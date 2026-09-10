"""Episodic memory - reward-tagged experience traces.

Backend note (ADR pending): dict-backed now; ChromaDB drop-in later.
Interface stability matters more than storage tech at this stage."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Episode:
    """One cognitive-cycle snapshot of experience."""
    position: tuple[int, int]
    action: int
    reward: float
    dopamine: float
    place_pattern: bytes          # serialized place-cell activation
    t_cycle: int = 0


class EpisodicMemory:
    """Append-only experience store with reward-ranked retrieval."""

    def __init__(self, capacity: int = 10_000) -> None:
        self.capacity = capacity
        self._episodes: list[Episode] = []
        self._t = 0

    def record(self, position: tuple[int, int], action: int,
               reward: float, dopamine: float,
               place_pattern: np.ndarray) -> None:
        ep = Episode(
            position=position, action=action, reward=reward,
            dopamine=dopamine, t_cycle=self._t,
            place_pattern=place_pattern.astype(np.uint8).tobytes(),
        )
        self._episodes.append(ep)
        self._t += 1
        if len(self._episodes) > self.capacity:
            self._episodes.pop(0)

    def top_rewarded(self, k: int = 50) -> list[Episode]:
        """Highest-reward episodes - the consolidation replay set."""
        return sorted(self._episodes, key=lambda e: e.reward, reverse=True)[:k]

    def at_position(self, position: tuple[int, int]) -> list[Episode]:
        return [e for e in self._episodes if e.position == position]

    def __len__(self) -> int:
        return len(self._episodes)


class WorkingMemory:
    """Baddeley-simplified: a capacity-limited buffer (7+-2) holding
    the current cognitive content with recency-based eviction."""

    def __init__(self, capacity: int = 7) -> None:
        self.capacity = capacity
        self._items: list[tuple[int, object]] = []   # (t, item)

    def push(self, item: object) -> None:
        self._items.append((len(self._items), item))
        if len(self._items) > self.capacity:
            self._items.pop(0)   # evict oldest - recency model

    def contents(self) -> list:
        return [item for _, item in self._items]

    def clear(self) -> None:
        self._items.clear()
