"""Mood - the slow valence baseline (ADR-012 third rung).

Timescale: tau in HOURS (default 3600s) vs emotions 2-30s vs dopamine
~100ms. Signed value (unlike Emotion, which floors at baseline):
mood drifts toward recent valence and SHIFTS the interpretation of
new events - a gloomy mood reads the same reward as less joyful.

Week 20 scope: the shift value exists and is consumed by callers
(engine-level integration is a one-line hook when needed)."""
from __future__ import annotations

import numpy as np


class Mood:
    def __init__(self, tau_s: float = 3600.0, dt_ms: float = 50.0) -> None:
        self.tau_ms = tau_s * 1000.0
        self.dt_ms = dt_ms
        self.valence = 0.0          # signed, -1..+1

    def step(self, current_valence: float) -> float:
        """Relax toward the recent valence stream (exponential)."""
        alpha = 1.0 - float(np.exp(-self.dt_ms / self.tau_ms))
        self.valence += alpha * (float(current_valence) - self.valence)
        self.valence = float(np.clip(self.valence, -1.0, 1.0))
        return self.valence

    def shift(self) -> float:
        """The bias applied to upcoming emotion interpretation."""
        return self.valence

    def reset(self) -> None:
        self.valence = 0.0
