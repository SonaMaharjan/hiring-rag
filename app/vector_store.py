import os
import chromadb
from typing import List, Dict, Any, Optional

DB_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "chroma_db"
)

# Initialize Chroma PersistentClient
_client = None

def get_chroma_client():
    global _client
    if _client is None:
        os.makedirs(DB_DIR, exist_ok=True)
        _client = chromadb.PersistentClient(path=DB_DIR)
    return _client

def get_collection(name: str):
    client = get_chroma_client()
    return client.get_or_create_collection(name=name)

# --- Profiles Helpers ---

def add_cv_profile(candidate_id: str, job_id: str, embedding: List[float]) -> None:
    collection = get_collection("cv_profiles")
    collection.add(
        ids=[candidate_id],
        embeddings=[embedding],
        metadatas=[{"candidate_id": candidate_id, "job_id": job_id}]
    )

def get_cv_profile_embedding(candidate_id: str, job_id: str) -> Optional[List[float]]:
    collection = get_collection("cv_profiles")
    result = collection.get(
        ids=[candidate_id],
        include=["embeddings"]
    )
    if result is not None and result.get("embeddings") is not None and len(result["embeddings"]) > 0:
        emb = result["embeddings"][0]
        if emb is not None and len(emb) > 0:
            return emb.tolist() if hasattr(emb, "tolist") else list(emb)
    return None

# --- Job Postings Vector Helpers ---

def add_job_posting_embedding(job_id: str, embedding: List[float]) -> None:
    collection = get_collection("job_postings")
    collection.add(
        ids=[job_id],
        embeddings=[embedding],
        metadatas=[{"job_id": job_id}]
    )

def get_job_posting_embedding(job_id: str) -> Optional[List[float]]:
    collection = get_collection("job_postings")
    result = collection.get(
        ids=[job_id],
        include=["embeddings"]
    )
    if result is not None and result.get("embeddings") is not None and len(result["embeddings"]) > 0:
        emb = result["embeddings"][0]
        if emb is not None and len(emb) > 0:
            return emb.tolist() if hasattr(emb, "tolist") else list(emb)
    return None

# --- Work Experience Helpers ---

def add_cv_experience_profile(candidate_id: str, job_id: str, embedding: List[float]) -> None:
    collection = get_collection("cv_experience_profiles")
    collection.add(
        ids=[candidate_id],
        embeddings=[embedding],
        metadatas=[{"candidate_id": candidate_id, "job_id": job_id}]
    )

def get_cv_experience_profile_embedding(candidate_id: str, job_id: str) -> Optional[List[float]]:
    collection = get_collection("cv_experience_profiles")
    result = collection.get(
        ids=[candidate_id],
        include=["embeddings"]
    )
    if result is not None and result.get("embeddings") is not None and len(result["embeddings"]) > 0:
        emb = result["embeddings"][0]
        if emb is not None and len(emb) > 0:
            return emb.tolist() if hasattr(emb, "tolist") else list(emb)
    return None

# --- Section Chunks Helpers ---

def add_cv_chunks(candidate_id: str, job_id: str, chunks: List[str], chunk_sections: List[str], embeddings: List[List[float]]) -> None:
    collection = get_collection("cv_chunks")
    ids = [f"{candidate_id}_chunk_{i}" for i in range(len(chunks))]
    metadatas = [
        {"candidate_id": candidate_id, "job_id": job_id, "section": section}
        for section in chunk_sections
    ]
    collection.add(
        ids=ids,
        embeddings=embeddings,
        metadatas=metadatas,
        documents=chunks
    )

def query_cv_chunks(candidate_id: str, query_embedding: List[float], k: int = 5) -> List[Dict[str, Any]]:
    collection = get_collection("cv_chunks")
    result = collection.query(
        query_embeddings=[query_embedding],
        n_results=k,
        where={"candidate_id": candidate_id}
    )
    
    formatted_results = []
    if result is not None and result.get("documents") is not None and len(result["documents"]) > 0:
        docs = result["documents"][0]
        metas = result["metadatas"][0] if result.get("metadatas") is not None and len(result["metadatas"]) > 0 else []
        distances = result.get("distances")[0] if result.get("distances") is not None and len(result["distances"]) > 0 else []
        
        for idx, doc in enumerate(docs):
            meta = metas[idx] if idx < len(metas) else {}
            formatted_results.append({
                "document": doc,
                "section": meta.get("section") if isinstance(meta, dict) else None,
                "distance": distances[idx] if idx < len(distances) else None
            })
            
    return formatted_results


# --- Deletion Helper ---

def delete_candidate_embeddings(candidate_id: str, job_id: str) -> None:
    """Removes all profile, experience, and chunk embeddings for a candidate."""
    try:
        # Delete from cv_profiles
        profiles_col = get_collection("cv_profiles")
        profiles_col.delete(ids=[candidate_id])
    except Exception as e:
        print(f"Error deleting from cv_profiles: {e}")
        
    try:
        # Delete from cv_experience_profiles
        exp_col = get_collection("cv_experience_profiles")
        exp_col.delete(ids=[candidate_id])
    except Exception as e:
        print(f"Error deleting from cv_experience_profiles: {e}")
        
    try:
        # Delete from cv_chunks
        chunks_col = get_collection("cv_chunks")
        chunks_col.delete(where={"candidate_id": candidate_id})
    except Exception as e:
        print(f"Error deleting from cv_chunks: {e}")
