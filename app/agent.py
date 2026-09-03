# ruff: noqa
# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from typing import Optional

from google.adk.agents import Agent
from google.adk.apps import App
from google.adk.models import Gemini, LiteLlm
from google.genai import types

from app import database
from app import pipelines
from app import scoring
from app import self_rag
from app import llm_client

# System instruction matching ReAct Agent requirements
SYSTEM_INSTRUCTION = (
    "You are an agentic Hiring RAG Assistant designed to help hiring managers rank and query candidates. "
    "Your workflow has two primary phases:\n"
    "1. **Job Ingestion & Approval Gate**: Managers submit job postings. You extract key requirements and "
    "MUST wait for the manager's explicit command containing the exact word 'approve' to lock in the job and "
    "ingest candidate resumes. You must block ranking operations if the job is not yet approved.\n"
    "2. **Ranking and Chatbot**: Once approved, managers can request candidate rankings and ask specific questions "
    "about candidates. You MUST answer candidate-specific details strictly from the CV corpus via Self-RAG. "
    "Never make up information or fall back to web searches.\n\n"
    "Always rely on your tools to retrieve information, rank candidates, or check requirements."
)

def submit_job_posting(posting_text: str, session_id: str = "default_session") -> str:
    """Ingests a new job posting text, extracts requirements, and sets status to pending approval.
    
    Args:
        posting_text: The complete text description of the job posting.
        session_id: Automatically injected session ID to track state.
        
    Returns:
        A formatted string showing the extracted requirements and prompting for approval.
    """
    try:
        job_id = pipelines.ingest_job_posting(posting_text)
        database.set_session_state(session_id, job_id=job_id)
        
        job_reqs = database.get_job_posting(job_id)
        if not job_reqs:
            return "Failed to ingest the job posting."
            
        return (
            f"### job_id extracted: `{job_id}`\n\n"
            f"**[PENDING APPROVAL] extracted Job Requirements:**\n"
            f"- **Education**: {job_reqs.get('education')}\n"
            f"- **Language**: {job_reqs.get('language')}\n"
            f"- **Certifications**: {job_reqs.get('certifications')}\n"
            f"- **Required Experience Years**: {job_reqs.get('required_years')} years\n\n"
            f"Please review these requirements. To confirm and proceed with candidate ingestion and ranking, "
            f"reply with the exact word: **approve**"
        )
    except Exception as e:
        return f"Error during job posting ingestion: {str(e)}"


def approve_job(confirmation: str, session_id: str = "default_session") -> str:
    """Approves the currently pending job posting and automatically ingests all resumes.
    
    Args:
        confirmation: Must contain the exact keyword 'approve' to proceed.
        session_id: Automatically injected session ID.
        
    Returns:
        A status message summarizing the approval and resume ingestion results.
    """
    state = database.get_session_state(session_id)
    job_id = state.get("job_id")
    if not job_id:
        return "No active job posting found in this session. Please submit a job posting first."
        
    job_reqs = database.get_job_posting(job_id)
    if not job_reqs:
        return f"Job posting {job_id} not found."
        
    if job_reqs.get("approved") == 1:
        return f"Job posting {job_id} is already approved and ready."

    if "approve" not in confirmation.strip().lower():
        return "Approval failed. You must type the exact word 'approve' to proceed."

    # Approve in SQLite
    database.approve_job_posting(job_id)

    # Automatically scan resumes folder and ingest candidate PDFs
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    resumes_dir = os.path.join(project_root, "resumes")
    if not os.path.exists(resumes_dir):
        return f"Job approved successfully, but resumes directory '{resumes_dir}' does not exist to process candidates."

    pdf_files = [f for f in os.listdir(resumes_dir) if f.endswith('.pdf')]
    max_resumes = int(os.environ.get("MAX_RESUMES_TO_PROCESS", "0"))
    if max_resumes > 0:
        pdf_files = pdf_files[:max_resumes]
    success_count = 0
    injection_count = 0
    failed_count = 0
    injection_details = []

    for filename in pdf_files:
        pdf_path = os.path.join(resumes_dir, filename)
        try:
            pipelines.ingest_cv(pdf_path, job_id)
            success_count += 1
        except pipelines.PromptInjectionDetected as e:
            injection_count += 1
            injection_details.append(filename)
        except Exception as e:
            print(f"Error ingesting CV {filename}: {e}")
            failed_count += 1

    # Pre-calculate and cache ranks
    if success_count > 0:
        try:
            scoring.rank_and_cache_candidates(job_id)
        except Exception as e:
            print(f"Error pre-ranking candidates: {e}")

    result_msg = (
        f"### Job `{job_id}` successfully APPROVED!\n\n"
        f"Processed **{len(pdf_files)}** CVs in the resumes directory:\n"
        f"- Successfully Ingested: **{success_count}** candidates\n"
        f"- Flagged for Prompt Injection: **{injection_count}** candidates (isolated in `Prompt Injection Flag/` folder)\n"
    )
    if failed_count > 0:
        result_msg += f"- Failed parsing: **{failed_count}** candidates\n"
        
    if injection_count > 0:
        result_msg += f"\n> ⚠️ **Security Warning**: The following files were flagged for prompt injection and bypassed: {', '.join(injection_details)}"

    result_msg += "\n\nCandidate profiles and experience embeddings are cached. You can now run `rank_candidates`."
    return result_msg


def rank_candidates(top_n: int = 5, job_id: Optional[str] = None, session_id: str = "default_session") -> str:
    """Retrieves ranked candidates for a job grouped into 3 flag tiers.
    
    Args:
        top_n: Number of distinct ranks to return per tier (default 5).
        job_id: Optional. The job ID to rank. If omitted, uses the active job in the session.
        session_id: Automatically injected session ID.
        
    Returns:
        A Markdown formatted table showing the ranked candidates by tier.
    """
    if not job_id:
        state = database.get_session_state(session_id)
        job_id = state.get("job_id")
        
    if not job_id:
        return "No active job ID found. Please submit and approve a job posting, or specify a job_id."

    job_reqs = database.get_job_posting(job_id)
    if not job_reqs:
        return f"Job posting {job_id} not found."

    if not job_reqs.get("approved"):
        return f"Job posting `{job_id}` is NOT approved yet. Ranking is blocked. Please approve the job requirements first."

    # Update session state with job_id
    database.set_session_state(session_id, job_id=job_id)

    try:
        ranked_data = scoring.get_ranked_candidates(job_id, top_n=top_n)
        
        output = f"## Ranks for Job `{job_id}` (Top {top_n} distinct ranks per tier)\n\n"
        
        tier_names = {
            0: "Tier 0 (0 flags - Perfect gate pass)",
            1: "Tier 1 (1 flag - Minor gate mismatch)",
            2: "Tier 2 (2+ flags - Major gate mismatches)"
        }
        
        for tier in [0, 1, 2]:
            output += f"### {tier_names[tier]}\n"
            tier_list = ranked_data.get(tier, [])
            if not tier_list:
                output += "No candidates in this tier.\n\n"
                continue
                
            output += "| Rank | Score | Candidate ID | Name | Flags |\n"
            output += "|---|---|---|---|---|\n"
            for cand in tier_list:
                flags_str = ", ".join(cand["flags"]) if cand["flags"] else "None"
                output += f"| {cand['rank']} | {cand['composite_score']:.2f} | `{cand['candidate_id']}` | {cand['name']} | {flags_str} |\n"
            output += "\n"
            
        return output
    except Exception as e:
        return f"Error retrieving candidate ranks: {str(e)}"


def search_candidate(candidate_id: str, question: str, session_id: str = "default_session") -> str:
    """Queries details about a candidate's CV or explains their scoring.
    
    Args:
        candidate_id: The unique ID of the candidate (always explicitly provided).
        question: Free-text question about the candidate's CV or their ranking score.
        session_id: Automatically injected session ID.
        
    Returns:
        The response generated strictly from the candidate's CV (via Self-RAG) or the cached score explanation.
    """
    state = database.get_session_state(session_id)
    job_id = state.get("job_id")
    
    if not job_id:
        return "No active job ID found. Please rank candidates or set a job context first."

    # Update candidate context in session
    database.set_session_state(session_id, candidate_id=candidate_id)

    # Check if the query is explanatory
    is_explanatory = any(kw in question.lower() for kw in ["why", "score", "flag", "rank", "composite", "similarity"])
    
    if is_explanatory:
        # Explanatory Mode: fetch cached scores from SQLite
        score_data = database.get_cached_candidate_score(job_id, candidate_id)
        if not score_data:
            return f"No score profile found for candidate `{candidate_id}` and job `{job_id}`."
            
        cand_info = database.get_candidate(candidate_id)
        name = cand_info["name"] if cand_info else "Candidate"
        
        flags_str = "\n".join([f"- {flag}" for flag in score_data["flags"]]) if score_data["flags"] else "- None"
        
        explanation = (
            f"### Score Analysis for {name} (`{candidate_id}`)\n\n"
            f"**Composite Score**: `{score_data['composite_score']:.2f}` (Flag Tier: `{score_data['flag_tier']}`)\n\n"
            f"**Scoring Components**:\n"
            f"- **Similarity Score (30% Full CV / 70% Work Experience)**: `{score_data['similarity_score']:.2f}`\n"
            f"- **Skill Overlap Score (LLM judged)**: `{score_data['skill_overlap_score']:.2f}`\n"
            f"- **Experience Score (Ratio of required experience)**: `{score_data['experience_score']:.2f}`\n\n"
            f"**Fired Hard-Gate Flags**:\n{flags_str}"
        )
        return explanation
    else:
        # Literal Lookup Mode: Run Self-RAG
        try:
            return self_rag.query_candidate_cv_self_rag(candidate_id, job_id, question)
        except Exception as e:
            return f"Error during Self-RAG retrieval: {str(e)}"


# Hybrid Setup: Chat Agent runs on Gemini Cloud (1 request per chat message, perfect tool calling)
# Heavy pipelines (ingestion, audits, scoring, Self-RAG) run locally via Ollama in llm_client
agent_model = Gemini(
    model="gemini-2.5-flash",
    retry_options=types.HttpRetryOptions(attempts=3),
)

root_agent = Agent(
    name="root_agent",
    model=agent_model,
    instruction=SYSTEM_INSTRUCTION,
    tools=[submit_job_posting, approve_job, rank_candidates, search_candidate],
)

app = App(
    root_agent=root_agent,
    name="app",
)
