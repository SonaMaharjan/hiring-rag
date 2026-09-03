import os
import json
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

# Check environment toggles
USE_LOCAL_LLM = os.environ.get("USE_LOCAL_LLM", "false").lower() in ("true", "1", "yes")
LOCAL_LLM_URL = os.environ.get("LOCAL_LLM_URL", "http://localhost:11434/v1")
LOCAL_LLM_MODEL = os.environ.get("LOCAL_LLM_MODEL", "llama3.2")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")

_openai_client = None

def get_openai_client():
    global _openai_client
    if _openai_client is None:
        from openai import OpenAI
        _openai_client = OpenAI(
            base_url=LOCAL_LLM_URL,
            api_key="ollama"  # Ollama doesn't require a real API key
        )
    return _openai_client

def get_gemini_client():
    from google import genai
    return genai.Client()

from langsmith import traceable

@traceable(name="LLM Text Generation")
def generate_text(
    prompt: str, 
    json_mode: bool = False, 
    model_override: Optional[str] = None
) -> str:
    """Generates text completion from either a local LLM (Ollama) or Gemini Cloud API."""
    if USE_LOCAL_LLM:
        try:
            client = get_openai_client()
            kwargs = {
                "model": model_override or LOCAL_LLM_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.1,
            }
            if json_mode:
                kwargs["response_format"] = {"type": "json_object"}

            response = client.chat.completions.create(**kwargs)
            return response.choices[0].message.content or ""
        except Exception as e:
            print(f"Local LLM ({LOCAL_LLM_MODEL}) request failed: {e}. Ensure Ollama is running (`ollama serve`).")
            # Fallback to Gemini if Gemini API key is configured
            if os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_GENAI_USE_VERTEXAI"):
                print("Falling back to Gemini API...")
                return _generate_gemini(prompt, json_mode, model_override)
            raise e
    else:
        return _generate_gemini(prompt, json_mode, model_override)

def _generate_gemini(prompt: str, json_mode: bool = False, model_override: Optional[str] = None) -> str:
    from google.genai import types
    client = get_gemini_client()
    config = types.GenerateContentConfig(response_mime_type="application/json") if json_mode else None
    
    response = client.models.generate_content(
        model=model_override or GEMINI_MODEL,
        contents=prompt,
        config=config
    )
    return (response.text or "").strip()
