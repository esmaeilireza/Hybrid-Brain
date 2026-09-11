# Cortex6 acceptance v4 - Week 21 close. Full rewrite after
# patch-on-patch corruption (indentation of block-based stimulus).
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pytest

from regions.cortex6 import Cortex6

PARAMS = {
    "simulation": {"dt_ms": 1.0},
    "neuron": {"tau_m_ms": 20.0, "v_rest_mV": -65.0,
               "v_threshold_mV": -55.0, "v_reset_mV": -70.0,
               "refractory_ms": 2.0},
    "synapse": {"tau_excite_ms": 5.0, "connection_density": 0.05,
                "excitatory_ratio": 0.8, "delay_min_ms": 1.0,
                "w_exc": 1.5, "w_inh": 3.0},
}


def _rates(c):
    totals = np.zeros(6)
    for _ in range(200):
        counts = c.step(np.full(c.n_in, 30.0))
        totals += np.array(counts, dtype=float)
    sizes = np.array([ly.n_exc for ly in c.layers], dtype=float)
    return (totals / 200.0 / sizes) * 1000.0


def test_all_layers_in_band():
    c = Cortex6(240, PARAMS, seed=1)
    for _ in range(50):
        c.step(np.full(c.n_in, 30.0))
    hz = _rates(c)
    assert all(2.0 < h < 120.0 for h in hz), f"rates={hz.round(1)}"


def test_gradient_decreases():
    c = Cortex6(240, PARAMS, seed=1)
    for _ in range(50):
        c.step(np.full(c.n_in, 30.0))
    hz = _rates(c)
    assert hz[0] > hz[5], f"no gradient: {hz.round(1)}"


# L5 tracking finding - re-measured Week 22 with the real
# MobileNet stream (documented in docs/benchmarks.md).
def test_output_tracks_stimulus():
    # Baseline-first design (friend's hypothesis: propagation lag
    # made the v1 OFF block measure the ON transient tail).
    c = Cortex6(240, PARAMS, seed=1)
    for _ in range(60):
        c.step(np.zeros(c.n_in))          # reach quiescence
    off = [c.step(np.zeros(c.n_in))[5] for _ in range(50)]
    for _ in range(40):
        c.step(np.full(c.n_in, 30.0))     # EMA + transient warm-up
    on = [c.step(np.full(c.n_in, 30.0))[5] for _ in range(50)]
    m_on, m_off = float(np.mean(on)), float(np.mean(off))
    r = (m_on - m_off) / max(m_on, 1e-9)
    assert r > 0.6, (f"contrast: {r:.2f} on={m_on:.1f} "
                      f"off={m_off:.1f}")



def test_stable_over_long_run():
    c = Cortex6(240, PARAMS, seed=1)
    for _ in range(50):
        c.step(np.full(c.n_in, 30.0))
    totals = np.zeros(6)
    for _ in range(1200):
        counts = c.step(np.full(c.n_in, 30.0))
        totals += np.array(counts, dtype=float)
    sizes = np.array([ly.n_exc for ly in c.layers], dtype=float)
    hz = totals / 1200.0 / sizes * 1000.0
    assert all(2.0 < h < 120.0 for h in hz), f"drifted: {hz.round(1)}"
