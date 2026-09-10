"""Attention - Cocktail-Party gain modulation (Week 13).

Design per homework (Week 12 review): attention amplifies INPUT GAIN
on the attended stream and raises its Workspace priority; settle
windows stay FIXED (lengthening them starves other regions and breaks
the fixed-horizon scheduling contract). A minimal dwell time (3
cycles) prevents flicker between streams.

Mechanism (Posner networks, simplified):
  - alerting: novelty/RPE dips raise the global gain ceiling
  - orienting: the stream with highest recent salience wins the gain
  - executive: PFC-class streams get priority weighting in the queue
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Stream:
    """One candidate input stream competing for attention."""
    name: str
    salience: float = 0.0
    gain: float = 1.0
    dwell_cycles: int = 0


class AttentionSystem:
    """Winner-take-most gain allocation with dwell-time anti-flicker."""

    def __init__(self, n_streams: int, base_gain: float = 1.0,
                 attended_gain: float = 3.0, min_dwell: int = 3,
                 seed: int = 0) -> None:
        self.streams = [Stream(f"stream_{i}") for i in range(n_streams)]
        self.base_gain = base_gain
        self.attended_gain = attended_gain
        self.min_dwell = min_dwell
        self.attended_idx = 0
        self._rng = np.random.default_rng(seed)

    def step(self, saliencies: list[float]) -> list[float]:
        """One cognitive cycle: re-allocate gains. Returns gain per stream."""
        for stream, sal in zip(self.streams, saliencies):
            stream.salience = float(sal)

        scores = [s.salience for s in self.streams]
        challenger = int(np.argmax(scores))
        current = self.streams[self.attended_idx]

        can_switch = current.dwell_cycles >= self.min_dwell
        clearly_better = scores[challenger] > scores[self.attended_idx] * 1.25
        if can_switch and (challenger != self.attended_idx and clearly_better):
            self.attended_idx = challenger
            self.streams[self.attended_idx].dwell_cycles = 0

        self.streams[self.attended_idx].dwell_cycles += 1

        gains = []
        for idx, stream in enumerate(self.streams):
            stream.gain = (self.attended_gain if idx == self.attended_idx
                           else self.base_gain)
            gains.append(stream.gain)
        return gains


class GlobalWorkspace:
    """Capacity-limited priority queue (Global Workspace, 7+-2 capacity).

    Regions submit items (name, payload, priority). Only the top
    `capacity` are broadcast this cycle; the rest are deferred."""

    def __init__(self, capacity: int = 7) -> None:
        self.capacity = capacity

    def broadcast(self, items):
        ranked = sorted(items, key=lambda t: t[2], reverse=True)
        return ranked[:self.capacity]
