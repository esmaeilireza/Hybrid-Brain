# Sweep Cortex6 v3 - ALL SIX layers printed; per-neuron Hz.
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
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
hdr = " ".join(f"L{i}Hz" for i in range(6))
print(f"{'w_ff':>6} {hdr}  verdict")
for w_ff in (40.0, 56.0, 72.0, 88.0, 104.0, 120.0):
    c = Cortex6(240, PARAMS, layer_density=0.10, w_ff=w_ff, seed=1)
    for _ in range(50):
        c.step(np.full(c.n_in, 30.0))
    totals = np.zeros(6)
    for _ in range(200):
        counts = c.step(np.full(c.n_in, 30.0))
        totals += np.array(counts, dtype=float)
    sizes = np.array([ly.n_exc for ly in c.layers], dtype=float)
    hz = totals / 200.0 / sizes * 1000.0
    ok = all(2.0 < h < 120.0 for h in hz)
    row = " ".join(f"{h:5.1f}" for h in hz)
    verdict_str = 'OK' if ok else 'out-of-band'
    print(f"{w_ff:6.1f} {row}  {verdict_str}")
