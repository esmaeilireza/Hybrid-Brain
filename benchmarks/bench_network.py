"""Month 1 headline deliverable: 10,000-neuron self-organizing network
throughput with STDP active. This number is the permanent baseline that
the Week-12 Rust port must be judged against."""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from core.neural_network import NeuralNetwork

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
    "stdp": {
        "tau_plus_ms": 20.0, "tau_minus_ms": 20.0,
        "a_plus": 0.01, "a_minus": 0.012, "w_max": 5.0,
    },
}


def main() -> None:
    n_exc, n_inh = 8_000, 2_000
    n_total = n_exc + n_inh
    print(f"Building network: {n_exc} E + {n_inh} I = {n_total} neurons...")
    t_build = time.perf_counter()
    net = NeuralNetwork(PARAMS, n_exc, n_inh, seed=42)
    print(f"Build time: {time.perf_counter() - t_build:.1f}s "
          f"(E->E alone has ~{int(0.05 * n_exc * n_exc):,} connections)")

    drive = np.full(n_exc, 12.0)
    n_ticks = 1_000
    total_spikes = 0

    print(f"Running {n_ticks} ticks with STDP active...")
    t0 = time.perf_counter()
    for _ in range(n_ticks):
        s_e, s_i = net.step(drive)
        total_spikes += int(s_e.sum()) + int(s_i.sum())
    elapsed = time.perf_counter() - t0

    throughput = n_total * n_ticks / elapsed
    mean_rate = total_spikes / (n_ticks / 1000.0) / n_total
    print(f"Elapsed: {elapsed:.2f}s for {n_total} neurons x {n_ticks} ticks")
    print(f"Throughput: {throughput:,.0f} neuron-ticks/s")
    print(f"Total spikes: {total_spikes:,} (mean rate {mean_rate:.1f} Hz)")

    bench_file = Path("docs/benchmarks.md")
    header = (
        "| Hardware | Backend | Neurons | Ticks | Throughput | Wall time | Mean rate |\n"
        "|----------|---------|---------|-------|------------|-----------|-----------|\n"
    )
    if not bench_file.exists():
        bench_file.write_text("# Hybrid-Brain - Benchmark Record\n\n" + header,
                              encoding="utf-8")
    with open(bench_file, "a", encoding="utf-8") as fh:
        fh.write(
            f"| i7-6700HQ / 24GB | Python+NumPy+scipy | {n_total} | {n_ticks} "
            f"| {throughput:,.0f} neuron-ticks/s | {elapsed:.2f}s | {mean_rate:.1f} Hz |\n"
        )
    print("Result appended to docs/benchmarks.md")


if __name__ == "__main__":
    main()
