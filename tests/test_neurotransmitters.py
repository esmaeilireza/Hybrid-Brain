"""Week 3 tests - neuromodulator dynamics."""
import numpy as np
import pytest

from core.neurotransmitters import NeuromodulatorSystem

PARAMS = {"simulation": {"dt_ms": 1.0}}


def test_levels_decay_to_baseline() -> None:
    sys_ = NeuromodulatorSystem(PARAMS)
    sys_.dopamine.release(1.0)
    level = 2.0
    for _ in range(5000):
        level = sys_.dopamine.step()
    assert abs(level - sys_.dopamine.baseline) < 0.01


def test_dopamine_rises_then_falls() -> None:
    sys_ = NeuromodulatorSystem(PARAMS)
    sys_.dopamine.release(0.5)
    peak = sys_.dopamine.level
    mid = sys_.dopamine.step()
    assert mid <= peak
    for _ in range(1000):
        mid = sys_.dopamine.step()
    assert mid < peak


def test_all_three_modulators_present() -> None:
    sys_ = NeuromodulatorSystem(PARAMS)
    out = sys_.step()
    assert set(out) == {"dopamine", "serotonin", "norepinephrine"}
