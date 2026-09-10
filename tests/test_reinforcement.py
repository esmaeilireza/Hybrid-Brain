"""Week 11 - RPE, dopamine gating, learned value selection."""
import numpy as np

from cognition.consolidation import ValueTable
from cognition.reinforcement import RPEAgent
from core.neurotransmitters import NeuromodulatorSystem

PARAMS = {"simulation": {"dt_ms": 1.0}}


def make_agent() -> RPEAgent:
    return RPEAgent(ValueTable(seed=1), NeuromodulatorSystem(PARAMS))


def test_rpe_positive_on_unexpected_reward() -> None:
    agent = make_agent()
    rpe = agent.compute_rpe((5, 5), 2, actual_reward=1.0)
    assert rpe > 0.5, "expected value should start near zero"


def test_rpe_shrinks_as_value_learns() -> None:
    agent = make_agent()
    rpes = []
    for _ in range(30):
        rpes.append(agent.learn((5, 5), 2, actual_reward=1.0))
    assert abs(rpes[-1]) < abs(rpes[0]) * 0.5, "value not converging"
    # Verified empirically (RPE printout, Week 11): single-state rule has
    # NO bootstrap -> monotone geometric convergence (0.986 * 0.5^n),
    # no overshoot. Overshoot signature moves to the trajectory harness
    # (accept_maze), where gamma*V(next_state) actually exists.
    assert rpes[0] > 0, "first encounter should be a positive surprise"
    assert all(rpes[i] >= rpes[i + 1] for i in range(len(rpes) - 1)), (
        "RPE not monotonically shrinking - convergence broken"
    )
    assert rpes[-1] < 1e-6, "value did not reach the reward magnitude"


def test_dopamine_tracks_rpe_sign() -> None:
    agent = make_agent()
    agent.learn((1, 1), 0, 1.0)          # positive RPE
    hi = agent.nt.dopamine.level
    for _ in range(5):
        agent.learn((2, 2), 0, -0.01)    # negative RPE repeatedly
    lo = agent.nt.dopamine.level
    assert hi > lo, "dopamine did not track RPE sign"


def test_value_signals_discriminable() -> None:
    agent = make_agent()
    for _ in range(20):
        agent.learn((3, 3), 1, 1.0)      # action 1 is good here
    v = agent.value_signals((3, 3))
    assert v[1] == v.max(), "learned best action not top-valued"
    assert v.max() - v.min() > 0.3, "signals too flat for BG threshold"
