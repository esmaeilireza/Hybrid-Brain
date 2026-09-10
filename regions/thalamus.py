"""Thalamus - sensory relay with sparse convergent afferents.

Design (fixed after test caught the dense-projection dilution bug):
each relay neuron reads kafferent sensor channels with equal weight.
A one-hot sensor input of 1.0 drives only the ~n*k/sensor_dim matched
neurons with current = gain/k (mV-scale) - enough to fire them - and
zero current to all others. Selectivity and drive come from sparsity,
not dilution."""
from __future__ import annotations

import numpy as np
from scipy import sparse

from core.neuron import LifPopulation


class Thalamus:
    def __init__(self, n_neurons: int, sensor_dim: int, params: dict,
                 gain: float = 250.0, k_afferents: int = 10,
                 seed: int = 0) -> None:
        self.pop = LifPopulation(n_neurons, params)
        self.gain = gain
        self.k = k_afferents
        rng = np.random.default_rng(seed)

        rows = np.repeat(np.arange(n_neurons), k_afferents)
        cols = rng.integers(0, sensor_dim, size=n_neurons * k_afferents)
        data = np.full(n_neurons * k_afferents, 1.0 / k_afferents)
        self.projection = sparse.csr_matrix(
            (data, (rows, cols)), shape=(n_neurons, sensor_dim)
        )

    def step(self, sensor_vector: np.ndarray) -> np.ndarray:
        current = self.gain * (self.projection @ sensor_vector)
        return self.pop.step(current)

    def reset(self) -> None:
        self.pop.reset()
