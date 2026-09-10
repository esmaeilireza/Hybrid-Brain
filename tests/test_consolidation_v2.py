"""Week 18 - ADR-005 redesign: damped, arousal-prioritized, capped replay."""
import numpy as np

from cognition.consolidation import Consolidation, ValueTable
from cognition.memory import EpisodicMemory


def make_setup():
    mem = EpisodicMemory()
    vt = ValueTable()
    return mem, vt, Consolidation(mem, vt)


def fake_pattern():
    return np.zeros(4, dtype=np.uint8)


def test_replay_strengthens_rewarded_action_damped() -> None:
    mem, vt, cons = make_setup()
    vt.reinforce((2, 2), 0, 1.0)                    # expected = 0.1
    mem.record((2, 2), 0, reward=1.0, dopamine=0.5,
               place_pattern=fake_pattern(), arousal=0.8)
    result = cons.sleep(replay_k=10)
    # damped replay: 1.0 * 0.1 * lr(0.1) = 0.01 added, not online-scale
    assert result["replayed"] == 1
    assert abs(vt.values_at((2, 2))[0] - 0.11) < 1e-9


def test_negative_rewards_replay_too() -> None:
    mem, vt, cons = make_setup()
    mem.record((5, 5), 1, reward=-0.9, dopamine=-0.3,
               place_pattern=fake_pattern(), arousal=0.9)
    result = cons.sleep(replay_k=10)
    assert result["replayed"] == 1
    assert vt.values_at((5, 5))[1] < 0.0            # punishment learned offline


def test_unrewarded_never_replays() -> None:
    mem, vt, cons = make_setup()
    mem.record((1, 1), 2, reward=0.0, dopamine=0.0,
               place_pattern=fake_pattern(), arousal=0.5)
    result = cons.sleep(replay_k=10)
    assert result["replayed"] == 0
    assert float(vt.values_at((1, 1))[2]) == 0.0


def test_per_cell_cap_bounds_total() -> None:
    mem, vt, cons = make_setup()
    cons.per_cell_cap = 0.1
    for _ in range(5):   # same (pos, action) repeatedly: 5 x 0.1 = 0.5 uncapped
        mem.record((3, 3), 0, reward=1.0, dopamine=0.5,
                   place_pattern=fake_pattern(), arousal=0.0)
    result = cons.sleep(replay_k=50)
    # capped: total delta per cell <= cap * lr = 0.1 * 0.1 = 0.01
    assert abs(vt.values_at((3, 3))[0] - 0.01) < 1e-9


def test_flashbulb_arousal_orders_replay() -> None:
    mem, vt, cons = make_setup()
    # low-arousal big reward vs high-arousal equal reward; k=1 slots
    mem.record((0, 0), 0, reward=1.0, dopamine=0.5,
               place_pattern=fake_pattern(), arousal=0.0)   # prio 1.0
    mem.record((9, 9), 1, reward=1.0, dopamine=0.5,
               place_pattern=fake_pattern(), arousal=1.0)   # prio 5.0
    result = cons.sleep(replay_k=1)
    assert result["replayed"] == 1
    assert float(vt.values_at((9, 9))[1]) != 0.0            # arousal won
    assert float(vt.values_at((0, 0))[0]) == 0.0            # low-arousal cut
