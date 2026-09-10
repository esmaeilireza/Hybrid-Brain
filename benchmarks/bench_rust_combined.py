"""Combined neurons+synapses speed benchmark, Python side.
Like-for-like: both sides run LIF (10k neurons) + one SynapseGroup
(500x500, 5% density) per tick. Rust combined number needs PyO3
(Week 15, ADR-008) - Python baseline recorded now."""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from core.neuron import LifPopulation
from core.synapse import SynapseGroup

N_NEURONS = 10_000
N_PRE, N_POST = 500, 500
N_TICKS = 200


def bench_python() -> float:
    pop = LifPopulation(N_NEURONS, {
        "simulation": {"dt_ms": 1.0},
        "neuron": {"tau_m_ms": 20.0, "v_rest_mV": -65.0,
                   "v_threshold_mV": -55.0, "v_reset_mV": -70.0,
                   "refractory_ms": 2.0},
    })
    params = {
        "simulation": {"dt_ms": 1.0, "seed": 42},
        "synapse": {"tau_excite_ms": 5.0, "connection_density": 0.05,
                    "excitatory_ratio": 0.8, "delay_min_ms": 1.0,
                    "w_exc": 1.5, "w_inh": 3.0},
    }
    syn = SynapseGroup(N_PRE, N_POST, params)
    rng = np.random.default_rng(42)
    spikes = (rng.random((N_TICKS, N_PRE)) < 0.05)

    t0 = time.perf_counter()
    for t in range(N_TICKS):
        pop.step(np.full(N_NEURONS, 15.0))
        syn.propagate(spikes[t].astype(float))
        syn.step()
    elapsed = time.perf_counter() - t0
    return N_NEURONS * N_TICKS / elapsed


def main() -> None:
    rate = bench_python()
    print(f"Python neurons+synapses : {rate:,.0f} neuron-ticks/s")
    with open("docs/benchmarks.md", "a", encoding="utf-8") as fh:
        fh.write(f"\n| Week 14 | neurons+synapses Python-only | "
                 f"{rate:,.0f} neuron-ticks/s "
                 f"(Rust combined pending PyO3, Week 15) |\n")
    print("Recorded in docs/benchmarks.md")


if __name__ == "__main__":
    main()