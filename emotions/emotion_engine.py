"""Emotion engine - six Ekman emotions, Valence-Arousal map,
and integration hooks (flashbulb, somatic markers, arousal gain).

Timescale hierarchy (ADR-012):
  dopamine:  tau ~100ms - transient credit assignment
  emotions:  tau 2-30s  - behavioral priors (this module)
  moods:     tau ~hours - slow baselines (emotions/mood.py, Week 20)

Update law: pure exponential relaxation level += (1-decay)*(target-level).
NO arbitrary damping factors: a multiplicative fudge (e.g. *0.5) makes
levels collapse to baseline independent of tau and INVERTS the hierarchy
- caught by test_timescale_hierarchy (regression tripwire, fired once).
"""
from __future__ import annotations

import numpy as np


class Emotion:
    """One scalar emotion with exponential rise/decay to baseline."""

    def __init__(self, name: str, baseline: float, tau_s: float,
                 dt_ms: float) -> None:
        self.name = name
        self.baseline = baseline
        self.level = baseline
        self.tau_ms = tau_s * 1000.0
        self.dt_ms = dt_ms

    def step(self, drive: float | None = None) -> float:
        if drive is None:
            drive = self.baseline
        target = max(self.baseline, drive)
        alpha = 1.0 - float(np.exp(-self.dt_ms / self.tau_ms))
        self.level += alpha * (target - self.level)
        self.level = float(np.clip(self.level, 0.0, 1.0))
        return self.level

    def burst(self, amount: float) -> None:
        self.level = float(np.clip(self.level + amount, 0.0, 1.0))

    def reset(self) -> None:
        self.level = self.baseline


class EmotionEngine:
    """Six Ekman emotions driven by existing project signals:
    joy/surprise <- RPE (Month 3); fear <- amygdala (Month 2);
    sadness/anger/disgust <- negative-RPE statistics."""

    def __init__(self, dt_ms: float = 50.0) -> None:
        self.dt_ms = dt_ms
        self.joy = Emotion("joy", 0.0, 5.0, dt_ms)
        self.surprise = Emotion("surprise", 0.0, 2.0, dt_ms)
        self.fear = Emotion("fear", 0.0, 10.0, dt_ms)
        self.sadness = Emotion("sadness", 0.0, 30.0, dt_ms)
        self.anger = Emotion("anger", 0.0, 15.0, dt_ms)
        self.disgust = Emotion("disgust", 0.0, 20.0, dt_ms)
        self._neg_streak = 0
        self._blocked_streak = 0
        self._neg_cell_visits: dict = {}
        self._last_valence = 0.0
        self._last_arousal = 0.0

    def step(self, rpe: float, amygdala_activation: float = 0.0,
             distance_reduced: bool = True,
             current_cell_value: float = 0.0,
             pos_key=None) -> dict:
        joy_drive = max(rpe, 0.0)
        fear_drive = amygdala_activation

        if rpe < 0:
            self._neg_streak += 1
        else:
            self._neg_streak = max(0, self._neg_streak - 2)
        sadness_drive = min(self._neg_streak / 20.0, 1.0)

        if rpe < 0 and not distance_reduced:
            self._blocked_streak += 1
        else:
            self._blocked_streak = max(0, self._blocked_streak - 1)
        anger_drive = min(self._blocked_streak / 15.0, 1.0)

        disgust_drive = 0.0
        if pos_key is not None and current_cell_value < -0.5:
            self._neg_cell_visits[pos_key] = \
                self._neg_cell_visits.get(pos_key, 0) + 1
            disgust_drive = min(self._neg_cell_visits[pos_key] / 10.0, 1.0)

        self.joy.step(joy_drive)
        self.fear.step(fear_drive)
        self.sadness.step(sadness_drive)
        self.anger.step(anger_drive)
        self.disgust.step(disgust_drive)
        if abs(rpe) > 0.3:
            self.surprise.burst(min(abs(rpe), 1.0) * 0.5)
        self.surprise.step()

        valence = (self.joy.level * 0.8 - self.fear.level * 0.9
                   - self.sadness.level * 0.7 - self.anger.level * 0.8
                   - self.disgust.level * 0.6)
        arousal = (self.fear.level * 0.9 + self.anger.level * 0.8
                   + self.joy.level * 0.5 + self.surprise.level * 0.7
                   - self.sadness.level * 0.4)
        self._last_valence = float(np.clip(valence, -1.0, 1.0))
        self._last_arousal = float(np.clip(arousal, 0.0, 1.0))

        return {
            "joy": self.joy.level, "surprise": self.surprise.level,
            "fear": self.fear.level, "sadness": self.sadness.level,
            "anger": self.anger.level, "disgust": self.disgust.level,
            "valence": self._last_valence, "arousal": self._last_arousal,
        }

    def somatic_marker(self) -> float:
        neg_valence = -min(0.0, self._last_valence)
        return float(np.clip(-self._last_arousal * neg_valence * 0.5,
                             -0.5, 0.0))

    def flashbulb_multiplier(self) -> float:
        return 1.0 + 4.0 * self._last_arousal

    def attended_gain_scale(self) -> float:
        return 1.0 + 0.5 * self._last_arousal

    def reset(self) -> None:
        for e in (self.joy, self.surprise, self.fear, self.sadness,
                  self.anger, self.disgust):
            e.reset()
        self._neg_streak = 0
        self._blocked_streak = 0
        self._neg_cell_visits.clear()
        self._last_valence = 0.0
        self._last_arousal = 0.0
