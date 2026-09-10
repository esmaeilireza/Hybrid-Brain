"""Neuromodulators - first-order dynamics toward baseline.

Dopamine is the critical one: it is the RPE (reward prediction error)
carrier that Week 11 reinforcement learning will read. Keeping it as a
simple level with exponential decay is enough for now and biologically
defensible as a first approximation.
"""
from __future__ import annotations

import numpy as np


class Neurotransmitter:
    """Single modulator: level decays toward baseline with time constant tau."""

    def __init__(self, name: str, baseline: float, tau_ms: float, dt_ms: float) -> None:
        self.name = name
        self.baseline = baseline
        self.tau_ms = tau_ms
        self.dt_ms = dt_ms
        self.level = baseline

    def step(self) -> float:
        self.level += (self.dt_ms / self.tau_ms) * (self.baseline - self.level)
        return self.level

    def release(self, amount: float) -> float:
        """Transient release (e.g., dopamine burst on unexpected reward)."""
        self.level += amount
        return self.level

    def reset(self) -> None:
        self.level = self.baseline


class NeuromodulatorSystem:
    """The three modulators used across the project."""

    def __init__(self, params: dict) -> None:
        nt = params.get("neurotransmitters", {})
        dt = float(params["simulation"]["dt_ms"])
        self.dopamine = Neurotransmitter(
            "dopamine", float(nt.get("dopamine_baseline", 0.1)),
            float(nt.get("dopamine_tau_ms", 100.0)), dt,
        )
        self.serotonin = Neurotransmitter(
            "serotonin", float(nt.get("serotonin_baseline", 0.1)),
            float(nt.get("serotonin_tau_ms", 200.0)), dt,
        )
        self.norepinephrine = Neurotransmitter(
            "norepinephrine", float(nt.get("norepi_baseline", 0.1)),
            float(nt.get("norepi_tau_ms", 150.0)), dt,
        )

    def step(self) -> dict[str, float]:
        return {
            "dopamine": self.dopamine.step(),
            "serotonin": self.serotonin.step(),
            "norepinephrine": self.norepinephrine.step(),
        }

    def reset(self) -> None:
        self.dopamine.reset()
        self.serotonin.reset()
        self.norepinephrine.reset()
