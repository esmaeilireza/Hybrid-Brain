"""Week 9 p2 - consolidation replay into value table."""
import numpy as np

from cognition.consolidation import Consolidation, ValueTable
from cognition.memory import EpisodicMemory


def test_replay_strengthens_rewarded_actions() -> None:
    mem = EpisodicMemory()
    for _ in range(30):
        mem.record((5, 5), action=2, reward=1.0, dopamine=0.5,
                   place_pattern=np.zeros(4))
    vt = ValueTable(seed=1)
    before = vt.values_at((5, 5))[2]
    Consolidation(mem, vt).sleep(replay_k=10)
    after = vt.values_at((5, 5))[2]
    assert after > before * 3, "replay did not strengthen values"


def test_unrewarded_episodes_not_replayed() -> None:
    mem = EpisodicMemory()
    for _ in range(20):
        mem.record((5, 5), action=1, reward=-0.01, dopamine=0.1,
                   place_pattern=np.zeros(4))
    vt = ValueTable(seed=1)
    before = vt.values_at((5, 5))[1]
    Consolidation(mem, vt).sleep()
    assert vt.values_at((5, 5))[1] == before, "negative-reward replayed?"


def test_value_table_positions_independent() -> None:
    vt = ValueTable(seed=1)
    vt.reinforce((0, 0), 1, 1.0)
    other = vt.values_at((19, 19))
    assert other.max() < 0.15, "values leaked across positions"
