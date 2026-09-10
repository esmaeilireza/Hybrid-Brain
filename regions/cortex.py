"""Cortex - single layer (six layers -> Month 6), reusing the proven
Week-3 E/I balance recipe. Receives thalamic spikes, projects to PFC.
No new tuning: the E/I parameters that passed the Week 3 benchmark."""
from __future__ import annotations

import numpy as np

from core.neuron import LifPopulation
from core.synapse import SynapseGroup


class Cortex:
    def __init__(self, n_exc: int, params: dict, seed: int = 0) -> None:
        n_inh = n_exc // 4
        self.pop_e = LifPopulation(n_exc, params)
        self.pop_i = LifPopulation(n_inh, params)
        rng = np.random.default_rng(seed)
        mk = lambda a, b, pol: SynapseGroup(a, b, params, rng, polarity=pol)
        self.e2e = mk(n_exc, n_exc, "excitatory")
        self.e2i = mk(n_exc, n_inh, "excitatory")
        self.i2e = mk(n_inh, n_exc, "inhibitory")
        self.n_exc = n_exc

    def step(self, thalamic_current: np.ndarray | None) -> np.ndarray:
        drive = thalamic_current if thalamic_current is not None else np.zeros(self.n_exc)
        s_e = self.pop_e.step(drive + self.e2e.g + self.i2e.g)
        self.pop_i.step(self.e2i.g)
        s_e_f = s_e.astype(float)
        s_i_f = self.pop_i.last_spikes.astype(float)
        self.e2e.propagate(s_e_f); self.e2i.propagate(s_e_f)
        self.i2e.propagate(s_i_f)
        self.e2e.step(); self.e2i.step(); self.i2e.step()
        return s_e
