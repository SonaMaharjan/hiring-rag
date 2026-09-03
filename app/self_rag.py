from typing import List, Dict, Any

from app import pipelines
from app import vector_store
from app import embeddings
from app import llm_client

from langsmith import traceable

MODEL_NAME = "gemini-2.5-flash"

@traceable(name="Self-RAG Relevance Evaluator")
def evaluate_chunk_relevance(chunk: str, question: str, client: Any = None) -> bool:
    """Evaluates if a chunk is relevant to the user's question using LLM."""
    prompt = (
        "You are a retriever evaluating if a section of a candidate's CV is relevant to answer a user's question.\n\n"
        f"Question: {question}\n"
        f"CV Chunk: \"\"\"\n{chunk}\n\"\"\"\n\n"
        "Evaluate the relevance. Respond with exactly either 'RELEVANT' or 'IRRELEVANT' followed by a colon and a reason."
    )
    
    try:
        resp = llm_client.generate_text(prompt).strip().upper()
        return resp.startswith("RELEVANT")
    except Exception as e:
        print(f"Error grading chunk relevance: {e}")
        return False

@traceable(name="Self-RAG Grounding Checker")
def check_grounding(context: str, answer: str, client: Any = None) -> bool:
    """Checks if the generated answer is fully supported and grounded by the context."""
    prompt = (
        "You are an auditor verifying if an answer is fully grounded in and supported by the provided CV context. "
        "Every claim in the answer must be supported by the context. If there is any hallucination or unsupported claim, "
        "the answer is UNGROUNDED.\n\n"
        f"CV Context:\n\"\"\"\n{context}\n\"\"\"\n\n"
        f"Generated Answer:\n\"\"\"\n{answer}\n\"\"\"\n\n"
        "Respond with exactly either 'GROUNDED' or 'UNGROUNDED' followed by a colon and a reason."
    )
    
    try:
        resp = llm_client.generate_text(prompt).strip().upper()
        return resp.startswith("GROUNDED")
    except Exception as e:
        print(f"Error checking answer grounding: {e}")
        return False

@traceable(name="Self-RAG Answer Generator")
def generate_answer_from_context(context: str, question: str, client: Any = None) -> str:
    """Generates an answer to the question based ONLY on the context."""
    prompt = (
        "You are a helpful HR assistant. Answer the user's question about the candidate using ONLY the provided CV context. "
        "If the context does not contain the answer, respond with exactly: 'I cannot find this information in the candidate's CV.'\n\n"
        f"CV Context:\n\"\"\"\n{context}\n\"\"\"\n\n"
        f"Question: {question}\n\n"
        "Answer:"
    )
    
    try:
        return llm_client.generate_text(prompt).strip()
    except Exception as e:
        print(f"Error generating answer: {e}")
        return "I cannot find this information in the candidate's CV."

@traceable(name="Self-RAG Query Workflow")
def query_candidate_cv_self_rag(candidate_id: str, job_id: str, question: str) -> str:
    """Implements the Self-RAG loop to answer candidate-specific queries."""
    # 1. Generate query embedding
    query_emb = embeddings.get_embedding(question)
    
    # 2. Retrieve top chunks from Chroma cv_chunks
    retrieved_chunks = vector_store.query_cv_chunks(candidate_id, query_emb, k=6)
    if not retrieved_chunks:
        return "I cannot find this information in the candidate's CV."
        
    # 3. Filter chunks by relevance grading
    relevant_texts = []
    for chunk_data in retrieved_chunks:
        doc = chunk_data["document"]
        if evaluate_chunk_relevance(doc, question):
            relevant_texts.append(doc)
            
    if not relevant_texts:
        return "I cannot find this information in the candidate's CV."
        
    # 4. Generate candidate answer from relevant context
    context = "\n---\n".join(relevant_texts)
    generated_answer = generate_answer_from_context(context, question)
    
    # Check if we generated the standard empty fallback message
    if "I cannot find this information" in generated_answer:
        return "I cannot find this information in the candidate's CV."
        
    # 5. Check grounding
    if check_grounding(context, generated_answer):
        return generated_answer
    else:
        print("Self-RAG Grounding check failed! Answer was ungrounded. Returning empty fallback.")
        return "I cannot find this information in the candidate's CV."

