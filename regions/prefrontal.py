"""Prefrontal cortex - working memory via stabilized recurrent loop.

Parameters are exposed because Month-2 tuning found them delicate:
the e2i recruitment density must give every I neuron many inputs
(density 0.10 x 200 E / 50 I = 0.4 inputs each -> 2/3 of I silent ->
runaway). Sweep harness: benchmarks/tune_pfc.py."""
from __future__ import annotations

import numpy as np

from core.neuron import LifPopulation
from core.synapse import SynapseGroup


class PrefrontalCortex:
    def __init__(self, n_exc: int, params: dict,
                 loop_density: float = 0.25, loop_w: float = 3.0,
                 recruit_density: float = 0.15,
                 clamp_density: float = 0.30, clamp_w: float = 4.0,
                 seed: int = 0) -> None:
        n_inh = max(1, n_exc // 4)
        self.pop_e = LifPopulation(n_exc, params)
        self.pop_i = LifPopulation(n_inh, params)
        self.n_exc = n_exc
        rng = np.random.default_rng(seed)

        def grp(n_pre, n_post, density, w_exc, w_inh, pol):
            syn = {**params["synapse"], "connection_density": density,
                   "w_exc": w_exc, "w_inh": w_inh}
            return SynapseGroup(n_pre, n_post,
                                {**params, "synapse": syn}, rng, polarity=pol)

        self.e2e = grp(n_exc, n_exc, loop_density, loop_w, 3.0, "excitatory")
        self.e2i = grp(n_exc, n_inh, recruit_density, 3.0, 3.0, "excitatory")
        self.i2e = grp(n_inh, n_exc, clamp_density, 3.0, clamp_w, "inhibitory")

    def step(self, cue_current: np.ndarray | None) -> np.ndarray:
        drive_e = np.zeros(self.n_exc)
        if cue_current is not None:
            drive_e = drive_e + cue_current
        s_e = self.pop_e.step(drive_e + self.e2e.g + self.i2e.g)
        self.pop_i.step(self.e2i.g)
        s_e_f = s_e.astype(float)
        s_i_f = self.pop_i.last_spikes.astype(float)
        self.e2e.propagate(s_e_f); self.e2i.propagate(s_e_f)
        self.i2e.propagate(s_i_f)
        self.e2e.step(); self.e2i.step(); self.i2e.step()
        return s_e
