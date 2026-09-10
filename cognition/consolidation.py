"""Consolidation - replay of reward-tagged episodes during 'sleep'.
SUSPENDED from acceptance path (ADR-005 pending): replay magnitude
overwhelmed online learning in Week-11 A/B. ValueTable retained as
the RL value store - now with deterministic zero init (uniform init
+ normalization faked confident signals on unvisited cells)."""
from __future__ import annotations

import numpy as np

from cognition.memory import EpisodicMemory


class ValueTable:
    def __init__(self, grid_size: int = 20, n_actions: int = 4,
                 lr: float = 0.1, seed: int = 0) -> None:
        self.grid = grid_size
        self.n_actions = n_actions
        self.lr = lr
        self.table: dict[int, np.ndarray] = {}

    def values_at(self, position: tuple[int, int]) -> np.ndarray:
        key = position[0] * self.grid + position[1]
        if key not in self.table:
            # deterministic: unvisited = zero preference. Discovery is
            # exploration's job, not fake normalized noise.
            self.table[key] = np.zeros(self.n_actions)
        return self.table[key]

    def reinforce(self, position: tuple[int, int], action: int,
                  amount: float) -> None:
        key = position[0] * self.grid + position[1]
        if key not in self.table:
            self.table[key] = np.zeros(self.n_actions)
        self.table[key][action] += self.lr * amount
        self.table[key] = np.clip(self.table[key], -2.0, 2.0)


class Consolidation:
    """Suspended - see docstring. Kept for Month-4 redesign."""

    def __init__(self, memory: EpisodicMemory, value_table: ValueTable) -> None:
        self.memory = memory
        self.values = value_table

    def sleep(self, replay_k: int = 50, grid_size: int = 20) -> dict:
        return {"suspended": True}
