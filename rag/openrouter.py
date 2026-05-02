"""
NexaAI — Universal LLM Client (OpenRouter & NVIDIA NIM)
======================================================
Detects provider based on API key:
- nvapi-... -> NVIDIA NIM (High Performance)
- sk-or-... -> OpenRouter (Universal)
"""

import os, json, base64, requests

# Endpoints
OPENROUTER_BASE = "https://openrouter.ai/api/v1/chat/completions"
NVIDIA_BASE     = "https://integrate.api.nvidia.com/v1/chat/completions"

# Model Mapping
MODELS = {
    "openrouter": {
        "text":    "qwen/qwen3-next-80b-a3b-instruct:free",
        "vision":  "meta-llama/llama-3.2-11b-vision-instruct:free",
        "fallback":"meta-llama/llama-3-8b-instruct:free"
    },
    "nvidia": {
        "text":    "meta/llama-3.1-405b-instruct",
        "vision":  "meta/llama-3.2-90b-vision-instruct", # High precision OCR
        "fallback":"meta/llama-3.1-70b-instruct"
    }
}

REQUEST_TIMEOUT = 45

def _get_config(api_key: str):
    """Detect provider and return base URL + models."""
    if api_key.startswith("nvapi-"):
        return NVIDIA_BASE, MODELS["nvidia"]
    return OPENROUTER_BASE, MODELS["openrouter"]

def _headers(api_key: str) -> dict:
    return {
        "Authorization": f"Bearer {api_key}",
        "Content-Type":  "application/json",
        "X-Title":       "NexaAI Universal RAG"
    }

def is_available(api_key: str) -> bool:
    if not api_key: return False
    clean = str(api_key).strip()
    return bool(clean and len(clean) > 10 and "your_api_key" not in clean)

def ask_llm(context: str, question: str, api_key: str, filename: str = "") -> str | None:
    if not is_available(api_key): return None
    
    base_url, cfg = _get_config(api_key)
    provider = "NVIDIA" if "nvidia" in base_url else "OpenRouter"
    doc_hint = f" from the document '{filename}'" if filename else ""
    
    system_prompt = (
        "You are a precise document Q&A assistant. Answer using ONLY the context provided. "
        "Be concise. If the answer is not in the context, say 'I could not find that information.' "
        "Return ONLY the value requested with zero conversational filler."
    )
    user_content = f"Context{doc_hint}:\n{context.strip()}\n\nQuestion: {question}"

    payload = {
        "model": cfg["text"],
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_content},
        ],
        "temperature": 0.1,
        "max_tokens": 1024,
    }

    try:
        r = requests.post(base_url, headers=_headers(api_key), json=payload, timeout=REQUEST_TIMEOUT)
        r.raise_for_status()
        ans = r.json()["choices"][0]["message"]["content"].strip()
        if ans:
            print(f"[{provider}] {cfg['text']} answered OK.")
            return ans
    except Exception as e:
        print(f"[{provider}] Primary error: {e} — trying fallback.")
        payload["model"] = cfg["fallback"]
        try:
            r = requests.post(base_url, headers=_headers(api_key), json=payload, timeout=REQUEST_TIMEOUT)
            return r.json()["choices"][0]["message"]["content"].strip()
        except Exception: return None

def ocr_image(image_bytes: bytes, api_key: str, page_num: int = 1) -> str:
    if not is_available(api_key): return ""
    
    base_url, cfg = _get_config(api_key)
    provider = "NVIDIA" if "nvidia" in base_url else "OpenRouter"
    b64 = base64.b64encode(image_bytes).decode("utf-8")

    payload = {
        "model": cfg["vision"],
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": "Extract ALL text from this document image exactly. Preserve structure. No filler."},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
            ],
        }],
        "temperature": 0.0,
        "max_tokens": 4096, # High limit for full page OCR
    }

    try:
        r = requests.post(base_url, headers=_headers(api_key), json=payload, timeout=90)
        r.raise_for_status()
        text = r.json()["choices"][0]["message"]["content"].strip()
        print(f"[{provider}] OCR Page {page_num} completed.")
        return text
    except Exception as e:
        print(f"[{provider}] OCR error (page {page_num}): {e}")
        return ""
