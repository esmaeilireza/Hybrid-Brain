"""Consolidation - sleep-phase replay (ADR-005 redesign, Week 18).

ADR-005 found: replay magnitude (~10000x online updates) dominated the
value table. The redesign, now implemented, has THREE constraints - one
per identified failure:

  1. DAMPED RATE: replay applies reward * replay_lr_scale (default 0.1x
     the online path). Replay revises; it never re-teaches.
  2. |REWARD| x FLASHBULB priority: both signs replay, arousal-weighted.
     (Valence-only replay was an optimistic bias - homework, Week 18.)
  3. SLEEP-ONLY: replay runs exclusively in explicit sleep() calls -
     a separate phase, never interleaved with online steps.

Plus ADR-005's cap: total reinforcement per (position, action) per
sleep is bounded (per_cell_cap), so one hot episode cannot own a cell.

ValueTable remains the pure RL store - replay writes through the same
reinforce() as online learning, at damped magnitude."""
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
    """ADR-005 redesign - see module docstring for the three constraints."""

    def __init__(self, memory: EpisodicMemory, value_table: ValueTable,
                 replay_lr_scale: float = 0.1,
                 per_cell_cap: float = 0.3) -> None:
        self.memory = memory
        self.values = value_table
        self.replay_lr_scale = replay_lr_scale
        self.per_cell_cap = per_cell_cap

    def sleep(self, replay_k: int = 50, grid_size: int = 20,
              flashbulb=None) -> dict:
        """One sleep cycle: replay top-priority episodes damped and capped.
        flashbulb: callable arousal -> multiplier (default 1 + 4*arousal,
        matching emotion_engine.flashbulb_multiplier)."""
        if flashbulb is None:
            flashbulb = lambda a: 1.0 + 4.0 * max(0.0, min(1.0, a))

        episodes = self.memory.replay_priority(replay_k, flashbulb)
        applied: dict[tuple[int, int], float] = {}
        replayed = 0
        total_delta = 0.0
        for ep in episodes:
            if ep.reward == 0.0:
                continue    # zero-priority guard: unrewarded never replays
            key = (ep.position, ep.action)
            used = applied.get(key, 0.0)
            amount = ep.reward * self.replay_lr_scale
            # cap the total |delta| per (position, action) per sleep
            room = self.per_cell_cap - abs(used)
            if room <= 0:
                continue
            if abs(amount) > room:
                amount = room * (1.0 if amount > 0 else -1.0)
            before = float(self.values.values_at(ep.position)[ep.action])
            self.values.reinforce(ep.position, ep.action, amount)
            after = float(self.values.values_at(ep.position)[ep.action])
            applied[key] = used + abs(amount)
            replayed += 1
            total_delta += (after - before)

        return {"suspended": False, "replayed": replayed,
                "total_value_delta": total_delta,
                "cells_touched": len(applied)}
