"""Rust vs Python speed benchmark - neurons-only, SAME SCOPE both sides.

Fairness note: the Week-4 Python baseline (74,642 neuron-ticks/s)
included STDP, which profiling showed was the dominant cost. This
benchmark measures neurons-only on BOTH sides so the comparison is
like-for-like. The Rust-vs-Python ratio here is the honest port-speed
number; the STDP port decision (ADR-008) uses it."""
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from core.neuron import LifPopulation

PROBE = Path(__file__).resolve().parent.parent / "rust-core" / "target" / "release" / "ab_probe.exe"

N_NEURONS = 10_000
N_TICKS = 1_000
PARAMS_LINE = "1.0 20.0 -65.0 -55.0 -70.0 2.0 10000"


def bench_python() -> float:
    pop = LifPopulation(N_NEURONS, {
        "simulation": {"dt_ms": 1.0},
        "neuron": {
            "tau_m_ms": 20.0, "v_rest_mV": -65.0,
            "v_threshold_mV": -55.0, "v_reset_mV": -70.0,
            "refractory_ms": 2.0,
        },
    })
    rng = np.random.default_rng(42)
    input_stream = np.clip(
        rng.uniform(0, 30, (N_TICKS, N_NEURONS)) + rng.normal(0, 3, (N_TICKS, N_NEURONS)),
        0.0, 60.0,
    )
    t0 = time.perf_counter()
    for t in range(N_TICKS):
        pop.step(input_stream[t])
    elapsed = time.perf_counter() - t0
    return N_NEURONS * N_TICKS / elapsed


def bench_rust() -> float:
    stdin_text = ("BENCH\n" + PARAMS_LINE + "\n" + str(N_TICKS) + "\n"
                  + " ".join(f"{x:.6f}" for x in np.full(N_NEURONS, 15.0)) + "\n")
    t0 = time.perf_counter()
    proc = subprocess.run([str(PROBE)], input=stdin_text,
                          capture_output=True, text=True, timeout=600)
    wall = time.perf_counter() - t0
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr)
    line = next(l for l in proc.stdout.splitlines() if l.startswith("ELAPSED"))
    elapsed = float(line.split()[1])
    _ = wall  # wall includes IO; elapsed is the internal Rust loop time
    return N_NEURONS * N_TICKS / elapsed


def main() -> None:
    py_rate = bench_python()
    print(f"Python neurons-only : {py_rate:,.0f} neuron-ticks/s")

    rs_rate = bench_rust()
    print(f"Rust   neurons-only : {rs_rate:,.0f} neuron-ticks/s")
    print(f"Speedup: {rs_rate / py_rate:.1f}x")

    with open("docs/benchmarks.md", "a", encoding="utf-8") as fh:
        fh.write(f"\n| Week 12-13 | neurons-only A/B | Python {py_rate:,.0f} "
                 f"vs Rust {rs_rate:,.0f} neuron-ticks/s | "
                 f"speedup {rs_rate / py_rate:.1f}x |\n")
    print("Recorded in docs/benchmarks.md")


if __name__ == "__main__":
    main()
