"""Official Month 1 benchmark — progress metric: neuron-ticks per second.

For now: no neural core, only NumPy baseline measurement.
After Week 4, this file will connect to the real neural_network.
"""
import time

import numpy as np


def main() -> None:
    n_neurons = 10_000
    state = np.random.randn(n_neurons)          # simulated membrane potential
    input_current = np.random.randn(n_neurons) * 0.1

    start = time.perf_counter()
    n_ticks = 1_000
    for _ in range(n_ticks):
        state = 0.95 * state + input_current    # simplified LIF dynamics
    elapsed = time.perf_counter() - start

    throughput = n_neurons * n_ticks / elapsed
    print(f"NumPy baseline: {throughput:,.0f} neuron-tick/s")
    print("Target after Week 4 (real code): record in docs/benchmarks.md")


if __name__ == "__main__":
    main()
