import os
import json
import urllib.request
import urllib.error
from typing import List, Union

HF_API_TOKEN = os.environ.get("HF_API_TOKEN") or os.environ.get("HUGGINGFACE_API_TOKEN")
MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
API_URL = f"https://api-inference.huggingface.co/pipeline/feature-extraction/{MODEL_ID}"

# Global local model cache for fallback
_local_model = None

def _get_local_model():
    global _local_model
    if _local_model is None:
        try:
            from sentence_transformers import SentenceTransformer
            _local_model = SentenceTransformer(MODEL_ID)
        except ImportError:
            raise ImportError(
                "Neither HF_API_TOKEN was provided nor is 'sentence-transformers' installed. "
                "Please set HF_API_TOKEN in your environment or install sentence-transformers."
            )
    return _local_model

def get_embedding(text: str) -> List[float]:
    """Generates an embedding vector for a single text string."""
    if not text:
        return [0.0] * 384  # Default dimension for all-MiniLM-L6-v2 is 384
        
    if HF_API_TOKEN:
        try:
            headers = {
                "Authorization": f"Bearer {HF_API_TOKEN}",
                "Content-Type": "application/json"
            }
            data = json.dumps({"inputs": [text]}).encode("utf-8")
            req = urllib.request.Request(API_URL, data=data, headers=headers, method="POST")
            
            with urllib.request.urlopen(req, timeout=10) as response:
                result = json.loads(response.read().decode("utf-8"))
                # The API returns a nested list: [[val1, val2, ...]]
                if isinstance(result, list) and len(result) > 0:
                    return result[0]
        except Exception as e:
            print(f"HuggingFace API embedding request failed: {e}. Falling back to local model.")
            
    # Fallback to local model execution
    model = _get_local_model()
    return model.encode(text).tolist()

def get_embeddings(texts: List[str]) -> List[List[float]]:
    """Generates embedding vectors for a list of text strings."""
    if not texts:
        return []
        
    if HF_API_TOKEN:
        try:
            headers = {
                "Authorization": f"Bearer {HF_API_TOKEN}",
                "Content-Type": "application/json"
            }
            data = json.dumps({"inputs": texts}).encode("utf-8")
            req = urllib.request.Request(API_URL, data=data, headers=headers, method="POST")
            
            with urllib.request.urlopen(req, timeout=15) as response:
                result = json.loads(response.read().decode("utf-8"))
                if isinstance(result, list) and len(result) == len(texts):
                    return result
        except Exception as e:
            print(f"HuggingFace API batch embedding request failed: {e}. Falling back to local model.")
            
    # Fallback to local model execution
    model = _get_local_model()
    return model.encode(texts).tolist()
