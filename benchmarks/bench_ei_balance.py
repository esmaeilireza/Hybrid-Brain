"""Week 3 acceptance (rev 2): cortical E/I circuit with explicit polarity.

Correct wiring: E cells excite E and I; I cells inhibit E.
Prints firing-rate distributions, not just means - population averages
can hide bimodal dynamics (a lesson from rev 1: mean 2.6 Hz I activity
was mostly silent neurons plus a few fast ones).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from core.neuron import LifPopulation
from core.synapse import SynapseGroup

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


def main() -> None:
    n_e, n_i = 800, 200
    pop_e = LifPopulation(n_e, PARAMS)
    pop_i = LifPopulation(n_i, PARAMS)

    rng = np.random.default_rng(42)
    e2e = SynapseGroup(n_e, n_e, PARAMS, rng, polarity="excitatory")
    e2i = SynapseGroup(n_e, n_i, PARAMS, rng, polarity="excitatory")
    i2e = SynapseGroup(n_i, n_e, PARAMS, rng, polarity="inhibitory")

    n_ticks = 2000
    drive = np.full(n_e, 15.0)
    e_rate = np.zeros(n_e)
    i_rate = np.zeros(n_i)

    for _ in range(n_ticks):
        s_e = pop_e.step(drive + e2e.g + i2e.g)   # i2e.g is already negative
        s_i = pop_i.step(e2i.g)
        e2e.propagate(s_e.astype(float))
        e2i.propagate(s_e.astype(float))
        i2e.propagate(s_i.astype(float))
        e2e.step(); e2i.step(); i2e.step()
        e_rate += s_e
        i_rate += s_i

    f_e = e_rate / (n_ticks / 1000.0)
    f_i = i_rate / (n_ticks / 1000.0)

    for name, f in (("E", f_e), ("I", f_i)):
        print(
            f"{name}: median {np.median(f):6.1f} Hz | "
            f"p10 {np.percentile(f, 10):6.1f} | p90 {np.percentile(f, 90):6.1f}"
        )
    print(f"Balance ratio I/E (medians): {np.median(f_i) / max(np.median(f_e), 1e-9):.2f}")

    # Acceptance: both populations ALIVE and heterogeneity bounded
    assert np.median(f_e) > 2.0, "E population silent - circuit dead"
    assert np.median(f_i) > 2.0, "I population silent - feed-forward broken"
    assert np.median(f_i) < 250.0, "I population runaway - no gating"
    assert np.percentile(f_e, 10) > 0.5, "most E neurons silent - check wiring"
    assert np.percentile(f_i, 10) > 0.5, "most I neurons silent - this was rev 1's hidden failure"
    print("ACCEPTED: E/I circuit alive, balanced, and homogeneous")


if __name__ == "__main__":
    main()
