import math
import json
from typing import List, Dict, Any, Tuple
from google.genai import types

from app import database
from app import vector_store
from app import pipelines
from app import llm_client
from app import embeddings


MODEL_NAME = "gemini-2.5-flash"

def cosine_similarity(v1: Any, v2: Any) -> float:
    """Computes cosine similarity between two vectors."""
    if v1 is None or v2 is None:
        return 0.0
    if len(v1) == 0 or len(v2) == 0 or len(v1) != len(v2):
        return 0.0
    dot_product = sum(a * b for a, b in zip(v1, v2))
    magnitude_v1 = math.sqrt(sum(a * a for a in v1))
    magnitude_v2 = math.sqrt(sum(b * b for b in v2))
    if magnitude_v1 == 0 or magnitude_v2 == 0:
        return 0.0
    return dot_product / (magnitude_v1 * magnitude_v2)

from langsmith import traceable

# --- Hard Gates / Flag Checker ---

@traceable(name="Hard Gate Audit LLM")
def check_hard_gates_llm(job_reqs: Dict[str, Any], candidate: Dict[str, Any]) -> List[str]:
    """Uses LLM to check for positive contradictions in Education, Language, and Certifications."""
    prompt = (
        "You are an HR auditor verifying if a candidate explicitly fails hard gate requirements. "
        "Evaluate the candidate's details against the job requirements for Education, Language, and Certifications.\n\n"
        "Rules:\n"
        "- A FLAG should be raised ONLY if there is an EXPLICIT CONTRADICTION (e.g., Job requires Spanish Fluent, candidate explicitly lists they speak only English/French; or Job requires Master's in CS, candidate only has high school diploma).\n"
        "- Do NOT raise a flag if the information is missing, absent, or 'not mentioned' on the CV. Only positive contradictions trigger a flag.\n"
        "- Educations match if they are in the same/similar field or equal/higher level.\n\n"
        f"Job Requirements:\n"
        f"- Education: {job_reqs.get('education')}\n"
        f"- Language: {job_reqs.get('language')}\n"
        f"- Certifications: {job_reqs.get('certifications')}\n\n"
        f"Candidate Credentials:\n"
        f"- Education: {candidate.get('education')}\n"
        f"- Language: {candidate.get('language')}\n"
        f"- Certifications/Licenses: {candidate.get('licenses')}\n\n"
        "Return the output as a valid JSON object containing a list of strings named \"flags\". "
        "Each flag should describe the contradiction (e.g., \"Candidate explicitly lacks the required PMP certification\"). "
        "If there are no contradictions, return an empty list: {\"flags\": []}"
    )

    try:
        res_text = llm_client.generate_text(prompt, json_mode=True)
        data = json.loads(res_text)
        return data.get("flags", [])
    except Exception as e:
        print(f"Error checking hard gates via LLM: {e}")
        return []

# --- Skill Overlap LLM Judge ---

@traceable(name="Skill Overlap LLM Judge")
def evaluate_skill_overlap(cv_text: str, posting_reqs_text: str) -> float:
    """Uses LLM to evaluate semantic skill overlap between job requirements and CV text from 0.0 to 1.0."""
    prompt = (
        "Evaluate the skill overlap between the job requirements and the candidate's CV text. "
        "Focus on semantic matching. If a technology/skill implies another (e.g. Django implying Python, "
        "React implying JavaScript), that is a valid match. Do not restrict to exact string matches.\n\n"
        f"Job Requirements:\n\"\"\"\n{posting_reqs_text}\n\"\"\"\n\n"
        f"Candidate CV Text:\n\"\"\"\n{cv_text}\n\"\"\"\n\n"
        "Assign a score between 0.0 (no skill overlap) and 1.0 (perfect skill overlap) based on how well "
        "the candidate's skills match the job requirements. "
        "Return the output as a valid JSON object matching this schema:\n"
        "{\n"
        "  \"score\": float\n"
        "}"
    )

    try:
        res_text = llm_client.generate_text(prompt, json_mode=True)
        data = json.loads(res_text)
        return float(data.get("score", 0.0))
    except Exception as e:
        print(f"Error evaluating skill overlap: {e}")
        return 0.0

# --- Compute and Cache All Scores ---

@traceable(name="Candidate Ranking & Scoring Pipeline")
def rank_and_cache_candidates(job_id: str) -> None:
    """Computes composite scores, flags, and tiers for all candidates and caches them in SQLite."""
    job_reqs = database.get_job_posting(job_id)
    if not job_reqs:
        raise ValueError(f"Job posting {job_id} not found.")
        
    if not job_reqs.get("approved"):
        raise ValueError(f"Job posting {job_id} is not approved yet.")

    candidates = database.get_candidates_for_job(job_id)
    if not candidates:
        print(f"No candidates found for job {job_id}.")
        return

    # 1. Retrieve job posting requirements embedding
    posting_emb = vector_store.get_job_posting_embedding(job_id)
    if posting_emb is None or len(posting_emb) == 0:
        # Create requirements description to embed
        reqs_desc = (
            f"Education: {job_reqs.get('education')}. "
            f"Language: {job_reqs.get('language')}. "
            f"Certifications: {job_reqs.get('certifications')}."
        )
        posting_emb = embeddings.get_embedding(reqs_desc)
        vector_store.add_job_posting_embedding(job_id, posting_emb)

    # 2. Extract job posting text representation for skill overlap judge
    posting_reqs_text = (
        f"Education required: {job_reqs.get('education')}\n"
        f"Language required: {job_reqs.get('language')}\n"
        f"Certifications required: {job_reqs.get('certifications')}\n"
        f"Experience required: {job_reqs.get('required_years')} years."
    )

    raw_similarities = []
    candidate_similarity_data = []

    # 3. First pass: Compute raw similarities for min-max normalization
    for cand in candidates:
        cand_id = cand["candidate_id"]
        
        # Cosine similarity for full CV
        full_cv_emb = vector_store.get_cv_profile_embedding(cand_id, job_id)
        s_full = cosine_similarity(full_cv_emb, posting_emb) if full_cv_emb is not None and len(full_cv_emb) > 0 else 0.0
        
        # Cosine similarity for work experience
        exp_emb = vector_store.get_cv_experience_profile_embedding(cand_id, job_id)
        s_exp = cosine_similarity(exp_emb, posting_emb) if exp_emb is not None and len(exp_emb) > 0 else 0.0
        
        # Weighted similarity
        s_raw = 0.30 * s_full + 0.70 * s_exp
        raw_similarities.append(s_raw)
        candidate_similarity_data.append((cand, s_raw))

    # Determine min/max for normalization
    min_sim = min(raw_similarities) if len(raw_similarities) > 0 else 0.0
    max_sim = max(raw_similarities) if len(raw_similarities) > 0 else 0.0
    sim_range = max_sim - min_sim


    # 4. Second pass: Compute skill overlap, experience, flags, and composite score
    for cand, s_raw in candidate_similarity_data:
        cand_id = cand["candidate_id"]
        
        # Normalize similarity
        if sim_range == 0:
            norm_sim = 1.0
        else:
            norm_sim = (s_raw - min_sim) / sim_range

        # Experience score
        req_years = int(job_reqs.get("required_years", 0))
        cand_years = int(cand.get("experience_years", 0))
        if req_years == 0:
            exp_score = 1.0
        else:
            exp_score = min(cand_years / req_years, 1.0)

        # Skill overlap score
        # Pass top 3 most relevant chunks to keep local LLM context fast & concise
        chunks_info = vector_store.query_cv_chunks(cand_id, posting_emb, k=3)
        cv_chunks_text = "\n".join([chunk["document"] for chunk in chunks_info]) if chunks_info else cand.get("work_experience_text", "")
        skill_score = evaluate_skill_overlap(cv_chunks_text, posting_reqs_text)

        # Composite score calculation
        composite = 0.35 * norm_sim + 0.40 * skill_score + 0.25 * exp_score

        # Hard gate flags
        flags = check_hard_gates_llm(job_reqs, cand)
        flag_count = len(flags)
        
        # Determine flag tier
        if flag_count == 0:
            flag_tier = 0
        elif flag_count == 1:
            flag_tier = 1
        else:
            flag_tier = 2

        # Cache in SQLite
        database.cache_scores(
            job_id=job_id,
            candidate_id=cand_id,
            similarity_score=norm_sim,
            skill_overlap_score=skill_score,
            experience_score=exp_score,
            composite_score=composite,
            flags=flags,
            flag_count=flag_count,
            flag_tier=flag_tier
        )

# --- Tie-Aware Ranking Output ---

def get_ranked_candidates(job_id: str, top_n: int = 5) -> Dict[int, List[Dict[str, Any]]]:
    """Returns ranked candidates grouped into 3 flag tiers (0, 1, 2+ flags)."""
    scores = database.get_cached_scores(job_id)
    if not scores:
        # Re-compute and cache if not present
        rank_and_cache_candidates(job_id)
        scores = database.get_cached_scores(job_id)

    # Group scores by tier
    tiers = {0: [], 1: [], 2: []}
    for score in scores:
        tier = score["flag_tier"]
        # Fetch candidate info to get name
        cand_info = database.get_candidate(score["candidate_id"])
        name = cand_info["name"] if cand_info else "Candidate"
        
        score_data = {
            "candidate_id": score["candidate_id"],
            "name": name,
            "composite_score": round(score["composite_score"], 2),
            "flags": score["flags"]
        }
        tiers[tier].append(score_data)

    # Sort each tier and apply tie-aware top-N ranking
    ranked_output = {}
    for tier, candidates in tiers.items():
        # Sort descending by composite_score
        candidates.sort(key=lambda x: (-x["composite_score"], x["candidate_id"]))
        
        ranked_tier = []
        current_rank = 0
        prev_score = -1.0
        distinct_ranks_count = 0
        
        for cand in candidates:
            score = cand["composite_score"]
            if score != prev_score:
                distinct_ranks_count += 1
                current_rank = distinct_ranks_count
                prev_score = score
                
            if distinct_ranks_count > top_n:
                break
                
            cand_ranked = cand.copy()
            cand_ranked["rank"] = current_rank
            ranked_tier.append(cand_ranked)
            
        ranked_output[tier] = ranked_tier

    return ranked_output
