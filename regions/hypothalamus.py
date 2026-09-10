"""Hypothalamus - biological needs as slow homeostatic variables.

Not spiking: these are minute-scale signals (energy, sleep pressure)
that modulate the fast systems. Output is a small modulation vector
consumed by motivation (Month 5) and by the consolidation 'sleep'
phase (Month 3). Simplicity over simulation - the biology here is
timescale-separated from the ms-scale core, and honoring that is
more faithful than forcing it into spikes."""
from __future__ import annotations

import numpy as np


class Hypothalamus:
    def __init__(self, energy_decay_per_min: float = 0.01,
                 sleep_rise_per_min: float = 0.02,
                 recovery_rate: float = 0.5) -> None:
        self.energy = 1.0
        self.sleep_pressure = 0.0
        self.energy_decay = energy_decay_per_min
        self.sleep_rise = sleep_rise_per_min
        self.recovery_rate = recovery_rate

    def tick(self, minutes_elapsed: float = 1.0) -> dict[str, float]:
        self.energy = max(0.0, self.energy - self.energy_decay * minutes_elapsed)
        self.sleep_pressure = min(1.0, self.sleep_pressure + self.sleep_rise * minutes_elapsed)
        return {"energy": self.energy, "sleep_pressure": self.sleep_pressure}

    def sleep(self, minutes: float = 30.0) -> dict[str, float]:
        """Sleep phase: recovers energy, discharges sleep pressure.
        (Month 3 plugs memory consolidation replay in here.)"""
        self.energy = min(1.0, self.energy + self.recovery_rate * minutes / 30.0)
        self.sleep_pressure = max(0.0, self.sleep_pressure - minutes / 30.0)
        return {"energy": self.energy, "sleep_pressure": self.sleep_pressure}

    def needs_attention(self) -> bool:
        """True when any need crosses its threshold - drives motivation."""
        return self.energy < 0.2 or self.sleep_pressure > 0.8
