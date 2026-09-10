"""Week 9 tests - episodic store and working memory."""
import numpy as np

from cognition.memory import EpisodicMemory, WorkingMemory


def test_episodic_ranking_by_reward() -> None:
    mem = EpisodicMemory()
    rng = np.random.default_rng(0)
    for i in range(100):
        mem.record((i % 20, i // 20), i % 4,
                   reward=float(rng.uniform(-0.1, 1.0)),
                   dopamine=0.1, place_pattern=np.zeros(200))
    top = mem.top_rewarded(k=10)
    rewards = [e.reward for e in top]
    assert min(rewards) > 0.9 * max(rewards), "ranking broken"
    assert len(top) == 10


def test_capacity_eviction() -> None:
    mem = EpisodicMemory(capacity=10)
    for i in range(50):
        mem.record((0, 0), 0, 0.0, 0.1, np.zeros(5))
    assert len(mem) == 10


def test_position_query() -> None:
    mem = EpisodicMemory()
    for i in range(20):
        mem.record((3, 4), 0, 0.5, 0.1, np.zeros(5))
    assert len(mem.at_position((3, 4))) == 20
    assert len(mem.at_position((9, 9))) == 0


def test_working_memory_capacity_7() -> None:
    wm = WorkingMemory()
    for i in range(12):
        wm.push(i)
    assert wm.contents() == list(range(5, 12)), "recency eviction wrong"
