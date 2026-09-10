"""Sweep PFC loop/clamp gain. Prints delay-period mean Hz per combo.
Pick any combo landing in (2, 120) Hz, set as defaults, run tests."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from regions.prefrontal import PrefrontalCortex

PARAMS = {
    "simulation": {"dt_ms": 1.0, "seed": 42},
    "neuron": {
        "tau_m_ms": 20.0, "v_rest_mV": -65.0, "v_threshold_mV": -55.0,
        "v_reset_mV": -70.0, "refractory_ms": 2.0,
    },
    "synapse": {
        "tau_excite_ms": 5.0, "connection_density": 0.05,
        "excitatory_ratio": 0.8, "delay_min_ms": 1.0,
        "w_exc": 1.5, "w_inh": 3.0,
    },
}

print(f"{'den':>5} {'loop_w':>7} {'clamp_w':>8} {'delay Hz':>9}  verdict")
for loop_density in (0.08, 0.12, 0.16, 0.20, 0.25):
 for loop_w in (2.0, 3.0):
    for clamp_w in (3.0, 4.0, 5.0, 6.0):
        pfc = PrefrontalCortex(200, PARAMS, loop_density=loop_density,
                               loop_w=loop_w, clamp_w=clamp_w, seed=1)
        for _ in range(50):
            pfc.step(np.full(200, 25.0))
        total = 0
        for _ in range(200):
            total += int(pfc.step(None).sum())
        hz = total / 200.0 / 200.0 * 1000.0
        verdict = "OK" if 2.0 < hz < 120.0 else ("starved" if hz <= 2 else "runaway")
        print(f"{loop_density:5.2f} {loop_w:7.1f} {clamp_w:8.1f} {hz:9.1f}  {verdict}")
