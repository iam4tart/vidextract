import json
import requests
import os

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "qwen3.5:9b"

def load_system_prompt() -> str:
    if os.path.exists("sys_prompt.txt"):
        with open("sys_prompt.txt") as f:
            return f.read().strip()
    
    fallback = """You are an assistant that scores objects.
        For each object, determine how re-engineerable it is on a scale of 0.0 to 1.0.
        Return only a JSON array with no explanation.
        Format: [{"label": "...", "eng_score": 0.0, "category": "...", "sub_category": "...", "complexity": 0.0, "reasoning": "..."}]"""
    return fallback

SYSTEM_PROMPT = load_system_prompt()

def score_objects(objects: list[dict], threshold: float=0.4) -> list[dict]:
    labels = [o["label"] for o in objects]
    user_msg = f"Score these objects:\n{json.dumps(labels)}"
    
    print(f"Sending {len(labels)} labels to Ollama...")
    
    response = requests.post(OLLAMA_URL, json={
        "model": MODEL,
        "system": load_system_prompt(),
        "prompt": user_msg,
        "stream": False
    }, timeout=120)
    response.raise_for_status()
    
    raw = response.json()["response"].strip()
    
    if