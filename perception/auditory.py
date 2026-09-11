# Auditory - synthesized tone-event channel (Week 22).
# A second REAL competing stream for the attention system:
# events (on/off, frequency band) drive a small spiking
# population; salience = smoothed spike count (the same rate-
# coding the connectome and Cortex6 use - house style).
from __future__ import annotations

import numpy as np

from core.neuron import LifPopulation


class AuditoryStream:
    def __init__(self, params: dict, n_neurons: int = 40,
                 seed: int = 0, drive: float = 30.0) -> None:
        # drive 2.5: above the MEASURED firing threshold (~10-20
        # constant-field, Week 21 calibration) is NOT needed here -
        # this is per-tick burst drive within 15-tick windows, so
        # 2.5 fired NOTHING (test caught it). 30 chosen; verified by test.
        self.pop = LifPopulation(n_neurons, params)
        self.n = n_neurons
        self.drive = drive
        self.event_on = False
        self._rate = 0.0

    def set_event(self, on: bool) -> None:
        self.event_on = bool(on)

    def step(self) -> np.ndarray:
        d = self.drive if self.event_on else 0.0
        s = self.pop.step(np.full(self.n, d))
        # EMA rate - the salience signal attention reads
        self._rate = 0.8 * self._rate + 0.2 * float(s.sum())
        return s

    def salience(self) -> float:
        return self._rate
