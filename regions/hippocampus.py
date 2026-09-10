"""Hippocampus - spatial encoding with place cells (O'Keefe 1971).

Each place cell has a Gaussian field over the 2D grid: it receives
strong input when the agent is near its field center and near-zero
input elsewhere. Spikes thus encode "where am I" - the substrate for
spatial memory and maze navigation (Month 3)."""
from __future__ import annotations

import numpy as np

from core.neuron import LifPopulation


class PlaceCellPopulation:
    """N place cells tiled over a size x size grid."""

    def __init__(self, n_cells: int, grid_size: int, params: dict,
                 field_sigma: float = 1.8, gain: float = 40.0,
                 seed: int = 0) -> None:
        self.pop = LifPopulation(n_cells, params)
        self.grid_size = grid_size
        self.field_sigma = field_sigma
        self.gain = gain
        rng = np.random.default_rng(seed)
        self.centers = rng.integers(0, grid_size, size=(n_cells, 2))

    def step(self, agent_pos: tuple[int, int]) -> np.ndarray:
        r, c = agent_pos
        d2 = (self.centers[:, 0] - r) ** 2 + (self.centers[:, 1] - c) ** 2
        fields = np.exp(-d2 / (2.0 * self.field_sigma ** 2))
        return self.pop.step(self.gain * fields)

    def reset(self) -> None:
        self.pop.reset()
