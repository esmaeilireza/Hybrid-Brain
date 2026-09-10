"""Basal ganglia - Go/No-Go action selection.

value_gain calibrated to the Week-2 f-I curve. decide() runs a settling
window (~30 ms) because a competition measured 1 tick after reset sees
only resting neurons (charge time ~7 ticks even at 36 mV drive)."""
from __future__ import annotations

import numpy as np

from core.neuron import LifPopulation


class BasalGanglia:
    def __init__(self, n_actions: int, params: dict,
                 channel_size: int = 50, cross_inhibition: float = 6.0,
                 value_gain: float = 40.0, seed: int = 0) -> None:
        self.n_actions = n_actions
        self.channel = channel_size
        self.pop = LifPopulation(n_actions * channel_size, params)
        self.cross_inhibition = cross_inhibition
        self.value_gain = value_gain

    def decide(self, value_signals: np.ndarray, settle_ticks: int = 30) -> int:
        """Reset, let the competition run settle_ticks, read the winner."""
        self.pop.reset()
        acc = np.zeros(self.n_actions)
        for _ in range(settle_ticks):
            drive = np.repeat(value_signals, self.channel) * self.value_gain
            s = self.pop.step(drive)
            acc += s.reshape(self.n_actions, self.channel).sum(axis=1)
        others = acc.sum() - acc
        gated = acc - self.cross_inhibition * others
        return int(np.argmax(gated))
