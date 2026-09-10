"""Consolidation tests - ADR-005 contract, post-redesign (Week 18).

The former xfail (replay strengthens rewarded actions) is now a REAL
test per ADR-005's exit condition: "re-enabled with the same original
assertion" - high-reward (pos,action) must outrank low-reward after
sleep. The former "unrewarded not replayed" split into two contracts:
zero-reward never replays (guard), and negative rewards replay DAMPED
(|RPE| priority is the redesign - valence-only replay was the bias).
"""
import numpy as np

import pytest

from cognition.consolidation import Consolidation, ValueTable
from cognition.memory import EpisodicMemory


def fake_pattern():
    return np.zeros(4, dtype=np.uint8)


def test_replay_strengthens_rewarded_actions() -> None:
    """ADR-005 exit condition - same original ranking assertion."""
    mem = EpisodicMemory()
    mem.record((2, 2), 0, reward=1.0, dopamine=0.5,
               place_pattern=fake_pattern(), arousal=0.5)
    mem.record((4, 4), 3, reward=0.1, dopamine=0.1,
               place_pattern=fake_pattern(), arousal=0.5)
    vt = ValueTable()
    Consolidation(mem, vt).sleep(replay_k=10)
    assert (float(vt.values_at((2, 2))[0])
            > float(vt.values_at((4, 4))[3])), "ranking lost in replay"


def test_zero_reward_never_replays() -> None:
    mem = EpisodicMemory()
    mem.record((5, 5), 1, reward=0.0, dopamine=0.0,
               place_pattern=fake_pattern(), arousal=0.8)
    vt = ValueTable()
    result = Consolidation(mem, vt).sleep(replay_k=10)
    assert result["replayed"] == 0
    assert float(vt.values_at((5, 5))[1]) == 0.0


def test_negative_rewards_replay_damped_and_bounded() -> None:
    """ADR-005: |RPE| priority means punishment replays too - damped.
    20 x (−0.01 x 0.1 scale x 0.1 lr) = −0.002, far from the cap."""
    mem = EpisodicMemory()
    for _ in range(20):
        mem.record((5, 5), action=1, reward=-0.01, dopamine=0.1,
                   place_pattern=fake_pattern())
    vt = ValueTable()
    before = float(vt.values_at((5, 5))[1])
    result = Consolidation(mem, vt).sleep()
    after = float(vt.values_at((5, 5))[1])
    assert after < before                       # punishment learned
    assert after > -0.01                        # DAMPED, not online-scale
    assert result["replayed"] == 20


def test_value_table_positions_independent() -> None:
    """Preserved from the original suite - untouched semantics."""
    mem = EpisodicMemory()
    mem.record((1, 1), 0, reward=1.0, dopamine=0.5,
               place_pattern=fake_pattern(), arousal=0.5)
    vt = ValueTable()
    Consolidation(mem, vt).sleep(replay_k=10)
    assert float(vt.values_at((9, 9))[0]) == 0.0   # other cells untouched
