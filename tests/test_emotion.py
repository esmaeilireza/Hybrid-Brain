"""Week 17 - emotion dynamics, timescale hierarchy, V-A map."""
from emotions.emotion_engine import EmotionEngine


def make_engine() -> EmotionEngine:
    return EmotionEngine()


def test_timescale_hierarchy_fear_outlasts_surprise() -> None:
    e = make_engine()
    e.fear.burst(0.8)
    e.surprise.burst(0.8)
    for _ in range(10):
        e.step(rpe=0.0)
    # tau_fear=10s vs tau_surprise=2s: 0.8*e^-0.05 > 0.8*e^-0.25
    assert e.fear.level > e.surprise.level


def test_joy_tracks_positive_rpe() -> None:
    e = make_engine()
    for _ in range(60):          # 3s at tau=5s -> ~0.76
        e.step(rpe=0.8)
    assert e.joy.level > 0.3
    assert e.joy.level > e.sadness.level


def test_fear_rises_from_amygdala_and_decays() -> None:
    e = make_engine()
    for _ in range(200):         # 10s at tau=10s -> ~0.51
        e.step(rpe=0.0, amygdala_activation=0.8)
    assert e.fear.level > 0.4
    for _ in range(400):         # 20s more -> 0.51*e^-2 ~ 0.07
        e.step(rpe=0.0)
    assert e.fear.level < 0.2


def test_valence_arousal_coordinates() -> None:
    e = make_engine()
    for _ in range(60):
        e.step(rpe=0.9)
    state = e.step(rpe=0.9)
    assert state["valence"] > 0

    e2 = make_engine()
    for _ in range(150):
        e2.step(rpe=-0.1, amygdala_activation=0.9)
    state = e2.step(rpe=-0.1, amygdala_activation=0.9)
    assert state["valence"] < 0
    assert state["arousal"] > 0.3


def test_somatic_marker_negative_when_afraid() -> None:
    e = make_engine()
    for _ in range(150):
        e.step(rpe=-0.05, amygdala_activation=0.9)
    assert e.somatic_marker() < 0.0
