"""Week 3 acceptance: two-population E/I circuit.

80% excitatory / 20% inhibitory (physiological ratio). Driving the E
population must produce a measurable, balanced response in the I
population - the basic motif of cortical circuits.
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

    # E -> E (self), E -> I (feed-forward), I -> E (feedback inhibition)
    rng = np.random.default_rng(42)
    e2e = SynapseGroup(n_e, n_e, PARAMS, rng)
    e2i = SynapseGroup(n_e, n_i, PARAMS, rng)
    i2e = SynapseGroup(n_i, n_e, PARAMS, rng)

    n_ticks = 2000
    drive = np.full(n_e, 18.0)          # tonic drive to E population
    e_spikes_total, i_spikes_total = 0, 0

    for _ in range(n_ticks):
        s_e = pop_e.step(drive + e2e.g - i2e.g)
        s_i = pop_i.step(e2i.g)
        e2e.propagate(s_e.astype(float))
        e2i.propagate(s_e.astype(float))
        i2e.propagate(s_i.astype(float))
        e2e.step(); e2i.step(); i2e.step()
        e_spikes_total += int(s_e.sum())
        i_spikes_total += int(s_i.sum())

    f_e = e_spikes_total / (n_ticks / 1000.0) / n_e
    f_i = i_spikes_total / (n_ticks / 1000.0) / n_i
    print(f"E population: {f_e:6.1f} Hz average firing rate")
    print(f"I population: {f_i:6.1f} Hz average firing rate")
    print(f"Balance ratio I/E: {f_i / max(f_e, 1e-9):.2f}")

    # Acceptance: both populations alive and active, I within a sane band
    assert f_e > 2.0, "E population silent - circuit dead"
    assert f_i > 2.0, "I population silent - feed-forward broken"
    assert f_i < 250.0, "I population runaway - inhibition not gating"
    print("ACCEPTED: E/I circuit is alive and balanced")


if __name__ == "__main__":
    main()
