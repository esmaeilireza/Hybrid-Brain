# Language adapters (ADR-018). RuleBased is fully deterministic;
# OllamaLocal is only exercised if the service responds (skip else).
from cognition.language import OllamaLocal, RuleBasedLanguage


STATE = {
    "position": (3, 4),
    "emotions": {"fear": 0.6, "joy": 0.1},
    "attended": "visual_cortex",
    "cortex_rate": 42.0,
    "auditory_rate": 0.0,
}


def test_rulebased_deterministic():
    r1 = RuleBasedLanguage().generate(STATE)
    r2 = RuleBasedLanguage().generate(STATE)
    assert r1 == r2
    assert "fear" in r1 and "visual_cortex" in r1


def test_rulebased_silent_when_calm():
    s = dict(STATE, emotions={"joy": 0.0}, auditory_rate=0.0)
    text = RuleBasedLanguage().generate(s)
    assert "feeling" not in text and "I hear" not in text


def test_ollama_skips_when_absent():
    import urllib.request
    try:
        urllib.request.urlopen("http://localhost:11434/api/tags",
                               timeout=2)
    except Exception:
        import pytest
        pytest.skip("Ollama not running")
    text = OllamaLocal().generate(STATE)
    assert isinstance(text, str) and len(text) > 0
