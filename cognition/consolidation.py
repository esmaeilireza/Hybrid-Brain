"""Consolidation - replay of reward-tagged episodes during 'sleep'.

Mechanism: each replayed episode's place pattern drives a plastic
position->action value table via a delta rule scaled by the episode's
reward. Top-rewarded episodes replay most; the value table then biases
the BG value signals in future episodes. This is the Month-3 bridge:
memory (episodic store) -> behavior (learned values)."""
from __future__ import annotations

import numpy as np

from cognition.memory import EpisodicMemory


class ValueTable:
    """Position -> action values. Sparse dict keyed by cell index
    (r * 20 + c), 4 actions each. Learns only from replay."""

    def __init__(self, grid_size: int = 20, n_actions: int = 4,
                 lr: float = 0.1, seed: int = 0) -> None:
        self.grid = grid_size
        self.n_actions = n_actions
        self.lr = lr
        self.table: dict[int, np.ndarray] = {}
        self.rng = np.random.default_rng(seed)

    def values_at(self, position: tuple[int, int]) -> np.ndarray:
        key = position[0] * self.grid + position[1]
        if key not in self.table:
            self.table[key] = self.rng.uniform(0.0, 0.1, self.n_actions)
        return self.table[key]

    def reinforce(self, position: tuple[int, int], action: int,
                  amount: float) -> None:
        key = position[0] * self.grid + position[1]
        if key not in self.table:
            self.table[key] = self.rng.uniform(0.0, 0.1, self.n_actions)
        self.table[key][action] += self.lr * amount
        self.table[key] = np.clip(self.table[key], -2.0, 2.0)


class Consolidation:
    """Sleep phase: replay top-rewarded episodes into the value table."""

    def __init__(self, memory: EpisodicMemory, value_table: ValueTable) -> None:
        self.memory = memory
        self.values = value_table

    def sleep(self, replay_k: int = 50, grid_size: int = 20) -> dict:
        """Replay the top-k rewarded episodes. Replay count per episode
        scales with its reward (strong experiences replay harder)."""
        episodes = self.memory.top_rewarded(k=replay_k)
        replays = 0
        for ep in episodes:
            pos = ep.position
            # decode action from the stored trace (action was recorded)
            weight = max(ep.reward, 0.0)
            if weight > 0:
                n = max(1, int(weight * 10))
                for _ in range(n):
                    self.values.reinforce(pos, ep.action, weight)
                    replays += 1
        return {"episodes_replayed": len(episodes), "total_replays": replays}
