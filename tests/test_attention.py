"""Week 13 - gain allocation, dwell anti-flicker, workspace queue."""
import numpy as np

from cognition.attention import AttentionSystem, GlobalWorkspace


def test_gain_favors_attended_stream() -> None:
    att = AttentionSystem(2, attended_gain=3.0, min_dwell=3)
    for _ in range(5):
        gains = att.step([0.9, 0.1])
    assert gains[0] == 3.0 and gains[1] == 1.0
    assert att.attended_idx == 0


def test_attention_switches_on_clear_salience() -> None:
    att = AttentionSystem(2, min_dwell=3)
    for _ in range(5):
        att.step([0.9, 0.1])
    for _ in range(5):
        gains = att.step([0.1, 0.9])
    assert att.attended_idx == 1 and gains[1] == 3.0


def test_dwell_blocks_flicker() -> None:
    att = AttentionSystem(2, min_dwell=5)
    att.step([0.9, 0.1])
    att.step([0.9, 0.1])
    att.attended_idx = 0
    att.step([0.9, 0.95])
    assert att.attended_idx == 0, "flickered inside dwell period"


def test_workspace_capacity_7() -> None:
    gw = GlobalWorkspace(capacity=7)
    items = [(f"r{i}", i, float(i)) for i in range(15)]
    broadcast = gw.broadcast(items)
    assert len(broadcast) == 7
    assert broadcast[0][0] == "r14"


def test_workspace_priority_respects_gain() -> None:
    gw = GlobalWorkspace(capacity=2)
    items = [("quiet", "a", 0.3), ("attended", "b", 0.3 * 3.0)]
    broadcast = gw.broadcast(items)
    assert broadcast[0][0] == "attended", "attention gain ignored in priority"
