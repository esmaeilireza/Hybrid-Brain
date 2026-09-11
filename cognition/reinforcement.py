"""Reinforcement learning - RPE dopamine pathway (Schultz 1997).

RPE = actual reward - expected reward(values_at(position)[action]).
Positive RPE: dopamine rises above baseline -> Go channel strengthens.
Negative RPE: dopamine dips below baseline -> the channel weakens.

Week 19 (ADR-015): optional Satiation seam. When satiation is set, the
dopamine response to REPEATED identical rewards at the same position is
attenuated (brain-side state - the environment's reward signal is
untouched). Default None = Month 3 behavior byte-identical.

This is the seam where learning replaces engineering: the BG's value
signals switch from the hand-written distance function to this table."""
from __future__ import annotations

import numpy as np

from cognition.consolidation import ValueTable
from core.neurotransmitters import NeuromodulatorSystem


class RPEAgent:
    """Ties: world positions -> ValueTable -> dopamine -> BG values."""

    def __init__(self, value_table, neuromodulators,
                 satiation=None) -> None:
        self.values = value_table
        self.nt = neuromodulators
        self.satiation = satiation
        self.last_rpe = 0.0

    def compute_rpe(self, position, action, actual_reward: float) -> float:
        expected = float(self.values.values_at(position)[action])
        self.last_rpe = actual_reward - expected
        return self.last_rpe

    def dopamine_update(self, pos=None) -> float:
        """Dopamine follows RPE sign. With satiation set, positive-RPE
        release is attenuated for repeated identical (pos, reward)
        events (ADR-015)."""
        if self.last_rpe > 0:
            amount = min(self.last_rpe, 1.0)
            if self.satiation is not None and pos is not None:
                amount *= self.satiation.attenuate(pos, self.last_rpe)
            self.nt.dopamine.release(amount)
        elif self.last_rpe < 0:
            self.nt.dopamine.level = max(
                0.0, self.nt.dopamine.level + self.last_rpe * 0.1
            )
        return self.nt.dopamine.level

    def learn(self, position, action, actual_reward: float,
              next_position=None) -> float:
        """Full cycle: RPE -> dopamine -> value update.
        next_position enables the TD bootstrap: target = r + gamma*max(V(s')).
        Bootstrapping is MANDATORY for maze credit assignment - without it
        only the goal cell ever learns (found in Month 3 acceptance).
        Terminal steps pass next_position=None (target = r)."""
        gamma = 0.9
        if next_position is not None:
            bootstrap = gamma * float(
                self.values.values_at(next_position).max())
        else:
            bootstrap = 0.0
        expected = float(self.values.values_at(position)[action])
        self.last_rpe = (actual_reward + bootstrap) - expected
        self.dopamine_update(pos=position)
        self.values.reinforce(position, action, self.last_rpe * 5.0)
        return self.last_rpe

    def value_signals(self, position):
        """Learned value signals for BG selection - replaces the
        hand-written distance function from Week 6."""
        v = self.values.values_at(position).copy()
        vmin, vmax = v.min(), v.max()
        if vmax > vmin:
            v = 0.2 + 0.7 * (v - vmin) / (vmax - vmin)
        return v
