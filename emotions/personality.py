"""Personality - the OCEAN vector as parameter injection (ADR-016).

A TraitVector derives concrete parameter values for mechanisms that
already exist: Curiosity, exploration epsilon, SomaticMarkerMap gain,
fear decay tau, negative-RPE reinforcement gain. No behavioral rules -
traits change tendencies, not actions.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TraitVector:
    """OCEAN in [0,1]. 0.5 = population-neutral."""
    openness: float = 0.5
    conscientiousness: float = 0.5
    extraversion: float = 0.5      # social weight: Month 6 placeholder
    agreeableness: float = 0.5
    neuroticism: float = 0.5

    # --- derived parameters (the injections) ---
    @property
    def curiosity_mult(self) -> float:
        """Scales Curiosity.bonus_max: 0.5x .. 2.5x."""
        return 0.5 + 3.0 * self.openness

    @property
    def epsilon_floor(self) -> float:
        """Exploration floor: high-C exploits (0.05), low-C roams (0.30)."""
        return 0.05 + 0.25 * (1.0 - self.conscientiousness)

    @property
    def marker_gain_mult(self) -> float:
        """Punishment sensitivity via somatic-marker accrual: 0.5x..1.5x."""
        return 0.5 + 1.0 * self.agreeableness

    @property
    def fear_tau_scale(self) -> float:
        """High-N fear persists longer: tau scaled 1x .. 3x."""
        return 1.0 + 2.0 * self.neuroticism

    @property
    def neg_rpe_gain(self) -> float:
        """High-N reinforces punishments harder: 1x .. 3x."""
        return 1.0 + 2.0 * self.neuroticism

    def label(self) -> str:
        n = "highN" if self.neuroticism >= 0.5 else "lowN"
        c = "highC" if self.conscientiousness >= 0.5 else "lowC"
        o = "highO" if self.openness >= 0.5 else "lowO"
        return f"{n}-{c}-{o}"
