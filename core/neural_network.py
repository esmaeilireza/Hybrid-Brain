"""NeuralNetwork - assembles populations, synapses, and plasticity
into a single tickable unit with an E/I cortical structure."""
from __future__ import annotations

import numpy as np
from scipy import sparse

from core.neuron import LifPopulation
from core.plasticity import STDP
from core.synapse import SynapseGroup


class NeuralNetwork:
    """Cortical-motif network: E and I populations with recurrent
    E->E, feed-forward E->I, and feedback I->E (plus I->I)."""

    def __init__(self, params: dict, n_exc: int, n_inh: int, seed: int = 0) -> None:
        self.params = params
        self.rng = np.random.default_rng(seed)
        self.pop_e = LifPopulation(n_exc, params)
        self.pop_i = LifPopulation(n_inh, params)

        mk = lambda n_pre, n_post, pol: SynapseGroup(
            n_pre, n_post, params, self.rng, polarity=pol
        )
        self.e2e = mk(n_exc, n_exc, "excitatory")
        self.e2i = mk(n_exc, n_inh, "excitatory")
        self.i2e = mk(n_inh, n_exc, "inhibitory")
        self.i2i = mk(n_inh, n_inh, "inhibitory")

        self.stdp_e2e = STDP(self.e2e, params)
        self.stdp_e2i = STDP(self.e2i, params)

    def step(self, drive_exc: np.ndarray, drive_inh: np.ndarray | None = None):
        """One tick. Returns (spikes_exc, spikes_inh) boolean masks."""
        drive_inh = drive_inh if drive_inh is not None else np.zeros(self.pop_i.n)

        s_e = self.pop_e.step(drive_exc + self.e2e.g + self.i2e.g)
        s_i = self.pop_i.step(drive_inh + self.e2i.g + self.i2i.g)

        s_e_f, s_i_f = s_e.astype(float), s_i.astype(float)
        self.e2e.propagate(s_e_f); self.e2i.propagate(s_e_f)
        self.i2e.propagate(s_i_f); self.i2i.propagate(s_i_f)
        for syn in (self.e2e, self.e2i, self.i2e, self.i2i):
            syn.step()

        self.stdp_e2e.step(s_e, s_e)
        self.stdp_e2i.step(s_e, s_i)
        return s_e, s_i
