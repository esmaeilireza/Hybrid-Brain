"""Reinforcement learning - RPE dopamine pathway (Schultz 1997).

RPE = actual reward - expected reward(values_at(position)[action]).
Positive RPE: dopamine rises above baseline -> Go channel for the
taken action strengthens. Negative RPE: dopamine dips below baseline
-> the channel weakens (the amygdala's punishment gate, same sign
convention, now driving action values instead of fear).

This is the seam where learning replaces engineering: the BG's value
signals switch from the hand-written distance function to this table."""
from __future__ import annotations

import numpy as np

from cognition.consolidation import ValueTable
from core.neurotransmitters import NeuromodulatorSystem


class RPEAgent:
    """Ties: world positions -> ValueTable -> dopamine -> BG values."""

    def __init__(self, value_table: ValueTable,
                 neuromodulators: NeuromodulatorSystem) -> None:
        self.values = value_table
        self.nt = neuromodulators
        self.last_rpe = 0.0

    def compute_rpe(self, position: tuple[int, int], action: int,
                    actual_reward: float) -> float:
        expected = float(self.values.values_at(position)[action])
        self.last_rpe = actual_reward - expected
        return self.last_rpe

    def dopamine_update(self) -> float:
        """Release dopamine proportional to positive RPE; withhold
        (dip toward baseline) on negative RPE. Returns new level."""
        if self.last_rpe > 0:
            self.nt.dopamine.release(min(self.last_rpe, 1.0))
        elif self.last_rpe < 0:
            self.nt.dopamine.level = max(
                0.0, self.nt.dopamine.level + self.last_rpe * 0.1
            )
        return self.nt.dopamine.level

    def learn(self, position: tuple[int, int], action: int,
              actual_reward: float,
              next_position: tuple[int, int] | None = None) -> float:
        """Full cycle: RPE -> dopamine -> value update.
        next_position enables the TD bootstrap: target = r + gamma*max(V(s')).
        Bootstrapping is MANDATORY for maze credit assignment - without it
        only the goal cell ever learns (found in Month 3 acceptance).
        Terminal steps pass next_position=None (target = r)."""
        gamma = 0.9
        if next_position is not None:
            bootstrap = gamma * float(self.values.values_at(next_position).max())
        else:
            bootstrap = 0.0
        expected = float(self.values.values_at(position)[action])
        self.last_rpe = (actual_reward + bootstrap) - expected
        self.dopamine_update()
        self.values.reinforce(position, action, self.last_rpe * 5.0)
        return self.last_rpe

    def value_signals(self, position: tuple[int, int]) -> np.ndarray:
        """Learned value signals for BG selection - replaces the
        hand-written distance function from Week 6."""
        v = self.values.values_at(position).copy()
        vmin, vmax = v.min(), v.max()
        if vmax > vmin:
            v = 0.2 + 0.7 * (v - vmin) / (vmax - vmin)
        return v
