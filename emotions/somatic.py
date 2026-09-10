"""Somatic markers (Damasio) - position-keyed avoidance bias.

Read-time bias, never a write into the ValueTable: punishment + high
arousal at a position accrues a bounded penalty; the DecisionMaker adds
it to action utilities. Extinction: safe visits decay the penalty.

Bounded at -0.5 (matches emotion_engine.somatic_marker's bound - one
convention, documented in ADR-014).
"""
from __future__ import annotations


class SomaticMarkerMap:
    def __init__(self, max_penalty: float = -0.5, gain: float = 0.5,
                 extinction_factor: float = 0.7) -> None:
        self.max_penalty = max_penalty
        self.gain = gain
        self.extinction_factor = extinction_factor
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
        """Extinction: a visit without punishment decays the marker."""
        if pos in self._penalties:
            self._penalties[pos] *= self.extinction_factor
            if abs(self._penalties[pos]) < 0.01:
                del self._penalties[pos]
        return self._penalties.get(pos, 0.0)

    def penalty(self, pos: tuple) -> float:
        return self._penalties.get(pos, 0.0)

    def reset(self) -> None:
        self._penalties.clear()
