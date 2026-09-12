# narrate.py - live narration with hard fallback (meeting-safe).
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import json
import urllib.request

def narrate(beat_name: str, state: dict, model: str = "qwen2.5:3b") -> str:
    prompt = (
        "You are the inner cognitive voice of an artificial biological brain (Hy-Bra). "
        f"Current module: {beat_name}. Telemetry: {json.dumps(state, default=str)} "
        "In max 2 short scientifically grounded first-person sentences, describe your internal state. "
        "Do not mention being an AI."
    )
    try:
        body = json.dumps({
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.3}
        }).encode("utf-8")
        req = urllib.request.Request(
            "http://localhost:11434/api/generate",
            data=body,
            headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=5.0) as r:
            return json.loads(r.read().decode("utf-8")).get("response", "").strip()
    except Exception:
        try:
            from cognition.language import RuleBasedLanguage
            fallback_text = RuleBasedLanguage().generate(state)
        except Exception:
            fallback_text = "State acknowledged; adapting internal dynamics."
        return f"[narration skipped: LLM offline - RuleBased would say: {fallback_text}]"

if __name__ == "__main__":
    print(narrate("emotion dynamics", {"sadness": 0.17, "anger": 0.31, "position": (3, 4)}))
