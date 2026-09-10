"""Combined neurons+synapses through PyO3 vs Python baseline (64.99M).
PyO3 per-tick Vec conversion overhead is INCLUDED on the Rust side -
this measures realistic in-process integration, not the bare loop."""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

import hybrid_brain_core as hbc
from core.neuron import LifPopulation
from core.synapse import SynapseGroup

N_NEURONS = 10_000
N_PRE, N_POST = 500, 500
N_TICKS = 200


def build_python():
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
    return pop, syn


def build_rust():
    params = {
        "simulation": {"dt_ms": 1.0, "seed": 42},
        "synapse": {"tau_excite_ms": 5.0, "connection_density": 0.05,
                    "excitatory_ratio": 0.8, "delay_min_ms": 1.0,
                    "w_exc": 1.5, "w_inh": 3.0},
    }
    syn_py = SynapseGroup(N_PRE, N_POST, params)
    W = syn_py.W.tocsr()
    syn = hbc.PySyn(
        W.data.tolist(), [int(i) for i in W.indices],
        [int(i) for i in W.indptr], N_PRE, N_POST, 5.0, 1.0, 1,
    )
    pop = hbc.PyLif(N_NEURONS, 1.0, 20.0, -65.0, -55.0, -70.0, 2.0)
    return pop, syn


def bench(pop_step, syn_propagate, syn_step, drive, stream) -> float:
    t0 = time.perf_counter()
    for t in range(N_TICKS):
        pop_step(drive)
        syn_propagate(stream[t])
        syn_step()
    elapsed = time.perf_counter() - t0
    return N_NEURONS * N_TICKS / elapsed


def main() -> None:
    rng = np.random.default_rng(42)
    stream = (rng.random((N_TICKS, N_PRE)) < 0.05)
    drive = np.full(N_NEURONS, 15.0)

    pop, syn = build_python()
    py_rate = bench(pop.step, syn.propagate, syn.step, drive, stream)
    print(f"Python combined: {py_rate:,.0f} neuron-ticks/s")

    rpop, rsyn = build_rust()
    rs_rate = bench(rpop.step, rsyn.propagate, rsyn.step, drive, stream)
    print(f"Rust (PyO3) combined: {rs_rate:,.0f} neuron-ticks/s")
    print(f"Ratio: {rs_rate / py_rate:.2f}x")

    with open("docs/benchmarks.md", "a", encoding="utf-8") as fh:
        fh.write(f"\n| Week 15 | combined via PyO3 | Python {py_rate:,.0f} "
                 f"vs Rust {rs_rate:,.0f} | ratio {rs_rate / py_rate:.2f}x |\n")
    print("Recorded.")


if __name__ == "__main__":
    main()
