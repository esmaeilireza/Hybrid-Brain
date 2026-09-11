# Language - narration interface (ADR-018: narrator, never decider).
# LanguageModel protocol + two implementations:
#   RuleBasedLanguage - deterministic templates, zero deps, Pi-safe
#   OllamaLocal       - optional local LLM (http://localhost:11434)
# Both consume a read-only brain-state dict and emit narration.
from __future__ import annotations

import json
import urllib.request


class LanguageModel:
    def generate(self, state: dict) -> str:
        raise NotImplementedError


class RuleBasedLanguage(LanguageModel):
    # deterministic narration - no network, no model, always works
    def generate(self, state: dict) -> str:
        parts = []
        parts.append(f"position {state.get('position')}")
        emo = state.get("emotions", {})
        top = max(emo, key=emo.get) if emo else None
        if top and emo[top] > 0.05:
            parts.append(f"feeling {top} at {emo[top]:.2f}")
        att = state.get("attended")
        if att:
            parts.append(f"attending to {att}")
        vis = state.get("cortex_rate")
        if vis is not None:
            parts.append(f"visual activity {vis:.1f}")
        aud = state.get("auditory_rate")
        if aud is not None and aud > 0.1:
            parts.append("I hear something")
        return "; ".join(parts)


class OllamaLocal(LanguageModel):
    def __init__(self, model: str = "qwen2.5:3b",
                 url: str = "http://localhost:11434/api/generate",
                 timeout_s: float = 30.0) -> None:
        self.model = model
        self.url = url
        self.timeout_s = timeout_s

    def generate(self, state: dict) -> str:
        prompt = ("You narrate a simulated brain's state in one"
                  " first-person sentence. State: "
                  + json.dumps(state, default=str))
        body = json.dumps(
            {"model": self.model, "prompt": prompt, "stream": False}
        ).encode()
        req = urllib.request.Request(
            self.url, data=body,
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=self.timeout_s) as r:
            return json.loads(r.read()).get("response", "").strip()
