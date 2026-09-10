"""Spiking neurons - fully vectorized with NumPy.

Architecture rules for this file:
  1. Never loop over neurons in Python - all operations vectorized
  2. State lives in NumPy arrays, not Python objects
  3. Spike times via linear interpolation - sub-tick precision (ADR-003)
  4. All physiological values come from config - zero hardcoding
"""
from __future__ import annotations

import numpy as np


class LifPopulation:
    """Leaky Integrate-and-Fire population with sub-tick interpolation.

    Dynamics (explicit Euler):
        tau_m * dv/dt = -(v - v_rest) + I
        if v >= v_threshold -> spike, v = v_reset, refractory period

    Reset semantics (ADR-003 amendment): spike TIME keeps sub-tick
    precision in last_spike_frac (for STDP), but VOLTAGE resets to
    v_reset at the end of the tick (NEST/Brian2 convention).
    """

    def __init__(self, n_neurons: int, params: dict) -> None:
        if n_neurons <= 0:
            raise ValueError("n_neurons must be positive")

        self.n = n_neurons
        self.dt = float(params["simulation"]["dt_ms"])

        p = params["neuron"]
        self.tau_m = float(p["tau_m_ms"])
        self.v_rest = float(p["v_rest_mV"])
        self.v_threshold = float(p["v_threshold_mV"])
        self.v_reset = float(p["v_reset_mV"])
        self.refractory_steps = int(round(float(p["refractory_ms"]) / self.dt))

        # --- State: arrays, not objects ---
        self.v = np.full(n_neurons, self.v_rest, dtype=np.float64)
        self.refractory = np.zeros(n_neurons, dtype=np.int32)
        self.last_spike_frac = np.full(n_neurons, np.nan, dtype=np.float64)
        self.last_spikes = np.zeros(n_neurons, dtype=bool)

        # Leak coefficient: v_new = v + leak * (v_rest - v + I)
        self._leak = self.dt / self.tau_m

    def step(self, input_current: np.ndarray) -> np.ndarray:
        """Advance one tick. Returns boolean spike mask of this tick.

        input_current: effective current in mV-scale (R_m * I).
        """
        if input_current.shape != (self.n,):
            raise ValueError(f"input shape must be ({self.n},)")

        # 1) Voltage before update - needed for interpolation
        v_prev = self.v.copy()

        # 2) Explicit Euler - only neurons outside refractory period
        active = self.refractory <= 0
        self.v[active] = (
            v_prev[active]
            + self._leak * (self.v_rest - v_prev[active] + input_current[active])
        )

        # 3) Threshold crossing detection
        spiked_mask = active & (self.v >= self.v_threshold)

        # 4) Linear interpolation of exact spike time (ADR-003)
        self.last_spike_frac[:] = np.nan
        if spiked_mask.any():
            v_before = v_prev[spiked_mask]
            v_after = self.v[spiked_mask]
            denom = v_after - v_before
            crossed = v_before < self.v_threshold
            safe_denom = np.where(denom == 0.0, 1.0, denom)
            frac = np.where(
                crossed,
                np.clip((self.v_threshold - v_before) / safe_denom, 0.0, 1.0),
                0.0,
            )
            self.last_spike_frac[spiked_mask] = frac

            # 5) MANDATORY voltage reset - without this line the neuron
            #    stays above threshold and fires every refractory exit
            #    (the regression this test suite is designed to catch).
            self.v[spiked_mask] = self.v_reset

        # 6/7) Refractory: spikers get the FULL period; everyone else
        #      counts down. (Decrementing spikers in the same tick made
        #      the effective refractory 1 tick instead of 2 - this was
        #      the dominant source of the f-I validation bias.)
        self.refractory = np.where(
            spiked_mask,
            self.refractory_steps,
            np.maximum(self.refractory - 1, 0),
        )

        self.last_spikes = spiked_mask

        return spiked_mask

    def reset(self) -> None:
        self.v[:] = self.v_rest
        self.refractory[:] = 0
        self.last_spike_frac[:] = np.nan
        self.last_spikes[:] = False
