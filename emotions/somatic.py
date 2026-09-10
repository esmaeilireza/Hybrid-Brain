"""Somatic markers (Damasio) - position-keyed avoidance bias.

Read-time bias, never a write into the ValueTable: punishment + high
arousal at a position accrues a bounded penalty; the DecisionMaker adds
it to action utilities. Extinction: safe visits decay the penalty and
SNAP TO ZERO past a threshold - exponential decay is asymptotic, so
"extinct" must be a defined boundary, not an asymptote (and that
boundary is physiological: sub-threshold fear no longer biases action).

Extinction arithmetic (defaults): initial penalty max -0.5, decay x0.7,
threshold 0.02 ->
  after 9 visits:  0.5 * 0.7^9  = 0.0202  > 0.02 (still active)
  after 10 visits: 0.5 * 0.7^10 = 0.0141  < 0.02 -> snapped to 0.0
Bounded at -0.5 (matches emotion_engine.somatic_marker - one convention,
documented in ADR-014).
"""
from __future__ import annotations


class SomaticMarkerMap:
    def __init__(self, max_penalty: float = -0.5, gain: float = 0.5,
                 extinction_factor: float = 0.7,
                 extinction_threshold: float = 0.02) -> None:
        self.max_penalty = max_penalty
        self.gain = gain
        self.extinction_factor = extinction_factor
        self.extinction_threshold = extinction_threshold
        self._penalties: dict[tuple, float] = {}

    def record(self, pos: tuple, punishment: float,
               arousal: float) -> float:
        """Punishment (negative RPE) + arousal at pos -> penalty accrues.
        Reward events record nothing: markers come from harm, not joy."""
        if punishment >= 0 or arousal <= 0:
            return self.penalty(pos)
        delta = self.gain * arousal * min(abs(punishment), 1.0)
        new = max(self.max_penalty, self._penalties.get(pos, 0.0) - delta)
        self._penalties[pos] = float(new)
        return self._penalties[pos]

    def note_safe_visit(self, pos: tuple) -> float:
        """Extinction: a visit without punishment decays the marker,
        snapping to zero (key removed) past the threshold."""
        if pos in self._penalties:
            self._penalties[pos] *= self.extinction_factor
            if abs(self._penalties[pos]) < self.extinction_threshold:
                del self._penalties[pos]
        return self._penalties.get(pos, 0.0)

    def penalty(self, pos: tuple) -> float:
        return self._penalties.get(pos, 0.0)

    def reset(self) -> None:
        self._penalties.clear()
