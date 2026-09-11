"""Motivation - needs stack, curiosity, reward satiation (Week 19).

ADR-015 basis: satiation is BRAIN state, not world state. The
environment reports the world; the brain decides what it means.
Decaying the external reward would move the MDP under the learner's
feet and silently invalidate deterministic benchmarks; attenuating
the dopamine RESPONSE preserves the true value landscape.

Consumers/sources (all audited):
  - Hypothalamus.energy (Week 6, minutes-scale) -> energy deficit
  - amygdala activation (Week 18 chain)         -> safety need
  - visit counts (Month 3 novelty)              -> Curiosity
  - Satiation.attenuate -> RPEAgent optional seam
"""
from __future__ import annotations

import numpy as np


class Satiation:
    """Diminishing dopamine response to repeated identical rewards.

    Signature = (position, reward). Same signature repeated -> the
    response multiplier decays 1/(1 + k*repeats), floored at min_factor
    (never fully blind). A fresh reward elsewhere is at full strength."""

    def __init__(self, k: float = 0.5, min_factor: float = 0.2,
                 decay: float = 0.8) -> None:
        self.k = k
        self.min_factor = min_factor
        self.decay = decay            # per-step cooling (dataclass-era param)
        self._counts: dict[tuple, int] = {}

    def attenuate(self, pos: tuple, reward: float) -> float:
        """Record an event, return the response multiplier for it."""
        if reward <= 0.0:
            return 1.0
        key = (pos, round(float(reward), 6))
        n = self._counts.get(key, 0)
        self._counts[key] = n + 1
        return max(self.min_factor, 1.0 / (1.0 + self.k * n))

    def step(self) -> None:
        """Cooling: each step, one repeat of 'forgetting' per track."""
        for key in list(self._counts):
            self._counts[key] = max(0, self._counts[key] - 1)
            if self._counts[key] == 0:
                del self._counts[key]

    def peek(self, pos: tuple, reward: float) -> float:
        """Current multiplier WITHOUT recording."""
        key = (pos, round(float(reward), 6))
        n = self._counts.get(key, 0)
        return max(self.min_factor, 1.0 / (1.0 + self.k * n))

    def reset(self) -> None:
        self._counts.clear()


class Curiosity:
    """Intrinsic motivation: novelty bonus with a saturation curve.

    Formalizes the Month 3 entry-only novelty bonus as a first-class
    drive: bonus = bonus_max / (1 + visits / tau_visits). Saturates
    with familiarity, never negative, never exactly zero."""

    def __init__(self, bonus_max: float = 0.5,
                 tau_visits: float = 3.0) -> None:
        self.bonus_max = bonus_max
        self.tau_visits = tau_visits

    def bonus(self, visits: int) -> float:
        return float(self.bonus_max /
                     (1.0 + max(visits, 0) / self.tau_visits))

    def get_intrinsic_reward(self, visits: int) -> float:
        return self.bonus(visits)

    def reset(self) -> None:
        pass    # stateless bonus function - kept for API symmetry


class MotivationSystem:
    """Maslow-simplified: energy > safety > curiosity.

    The top UNMET need biases the BG value function: actions the
    caller tags as satisfying that need get their values multiplied
    (parameter injection, not if/else on behaviors). Safety's source
    is the amygdala (Week 18 chain) - the Hypothalamus has no safety
    variable, per its audited contract."""

    NEEDS = ("energy", "safety", "curiosity")

    def __init__(self, hypothalamus,
                 amygdala_activation: float = 0.0,
                 energy_threshold: float = 0.5,
                 safety_threshold: float = 0.3) -> None:
        self.hypo = hypothalamus
        self.amygdala_activation = float(amygdala_activation)
        self.energy_threshold = energy_threshold
        self.safety_threshold = safety_threshold

    def top_need(self) -> str:
        """Name of the top unmet need (curiosity is never 'met' -
        it saturates; a drive with a curve, not a threshold)."""
        if (1.0 - float(self.hypo.energy)) >= self.energy_threshold:
            return "energy"
        if self.amygdala_activation >= self.safety_threshold:
            return "safety"
        return "curiosity"

    def needs_bias(self, values,
                   satisfaction) -> np.ndarray:
        """values: per-action learned values. satisfaction: per-action
        0..1 estimate of how much each action serves the top need.
        Returns biased values (input untouched). Curiosity biases
        EXPLORATION (via Curiosity.bonus), not the argmax."""
        need = self.top_need()
        v = np.array(values, dtype=float)
        if need == "curiosity":
            return v
        s = np.clip(np.asarray(satisfaction, dtype=float)
                    [:len(v)], 0.0, 1.0)
        weight = {"energy": 0.8, "safety": 0.6}[need]
        return v * (1.0 + weight * s)


# Aliases (dataclass-era naming kept working)
CuriosityDrive = Curiosity
RewardSatiation = Satiation
NeedsStack = MotivationSystem
