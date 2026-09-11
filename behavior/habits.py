# Habits - cached S->A policies (ADR-019).
# Forms when RPE variance at a position is LOW (stable value table).
# Executes BALLISTICALLY: no value lookup, no BG call. An accumulated
# negative-RPE detector invalidates the cache (habit -> goal-directed
# control returns; the Week-14-23 dissociation).
from __future__ import annotations

import numpy as np


class Habits:
    def __init__(self, variance_threshold: float = 0.01,
                 invalidate_after: int = 3) -> None:
        self.variance_threshold = variance_threshold
        self.invalidate_after = invalidate_after
        self.cache: dict[tuple, int] = {}      # position -> action
        self._rpe_hist: dict[tuple, list] = {}
        self._miss_streak: dict[tuple, int] = {}
        self.executions = 0                    # benchmark stat

    def observe(self, pos: tuple, action: int, rpe: float) -> None:
        """Called every step by the RL loop with the CURRENT rpe."""
        h = self._rpe_hist.setdefault(pos, [])
        h.append(rpe)
        if len(h) > 20:
            h.pop(0)
        if len(h) >= 5 and np.var(h) < self.variance_threshold:
            if pos not in self.cache:
                self.cache[pos] = action

    def get(self, pos: tuple):
        """Cached action or None. On a cache MISS under an active
        habit, the miss-streak grows; too many misses = invalidated"""
        if pos in self.cache:
            self.executions += 1
            return self.cache[pos]
        return None

    def report_outcome(self, pos: tuple, rpe: float) -> None:
        """Habit execution feedback: repeated negative outcomes under
        a cached habit trip the invalidation detector (ADR-019)."""
        if pos not in self.cache:
            return
        if rpe < -0.05:
            s = self._miss_streak.get(pos, 0) + 1
            self._miss_streak[pos] = s
            if s >= self.invalidate_after:
                del self.cache[pos]
                self._miss_streak[pos] = 0
                self._rpe_hist.pop(pos, None)
        else:
            self._miss_streak[pos] = 0
