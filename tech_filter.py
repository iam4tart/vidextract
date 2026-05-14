import json
import requests
import os

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "qwen3:1.7b"

def load_system_prompt() -> str:
    if os.path.exists("sys_prompt.txt"):
        with open("sys_prompt.txt") as f:
            return f.read().strip()
    return """You are an assistant that scores objects.
For each object, determine how re-engineerable it is on a scale of 0.0 to 1.0.
Return only a JSON array with no explanation.
Format: [{"label": "...", "eng_score": 0.0, "category": "...", "sub_category": "...", "complexity": 0.0, "reasoning": "..."}]"""

def score_objects(objects: list[dict], threshold: float = 0.4) -> list[dict]:
    labels = [o["label"] for o in objects]
    user_msg = f"Score these objects:\n{json.dumps(labels)}"

    print(f"Sending {len(labels)} labels to Ollama...")

    response = requests.post(OLLAMA_URL, json={
        "model": MODEL,
        "system": load_system_prompt(),
        "prompt": user_msg,
        "stream": False,
        "options": {"temperature": 0}
    }, timeout=300)
    response.raise_for_status()

    raw = response.json()["response"].strip()

    if "<think>" in raw:
        raw = raw.split("</think>")[-1].strip()

    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    raw = raw.strip()

    scores = json.loads(raw)
    score_map = {s["label"].lower(): s for s in scores}

    curated = []
    for obj in objects:
        key = obj["label"].lower()
        s = score_map.get(key, {})
        eng_score = float(s.get("eng_score", 0.0))
        if eng_score >= threshold:
            curated.append({
                **obj,
                "eng_score": eng_score,
                "category": s.get("category", "none"),
                "sub_category": s.get("sub_category", "none"),
                "complexity": float(s.get("complexity", 0.0)),
                "reasoning": s.get("reasoning", "")
            })

    # high complexity objects show up first
    curated.sort(key=lambda x: -x["eng_score"])
    print(f"{len(curated)} objects passed the {threshold} threshold filter")
    return curated