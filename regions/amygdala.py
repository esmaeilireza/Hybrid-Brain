"""Amygdala - threat detection via dopamine-gated plasticity.

Conditioning protocol:
  Phase 1 (pre): place-cell -> amygdala synapses at baseline strength.
  Phase 2 (pairing): agent visits region R; punishment arrives through
    the reward pathway; STDP in the projecting synapses is gated ON
    only while punishment is present (dopamine < baseline).
  Phase 3 (test): amygdala fires at R with NO punishment - learned fear.

Gate convention: positive dopamine gates potentiation (reward learning),
negative-gated plasticity here uses the level dropping below baseline as
the "punishment present" signal - same pathway, opposite sign."""
from __future__ import annotations

import numpy as np
from scipy import sparse

from core.neuron import LifPopulation


class Amygdala:
    def __init__(self, n_neurons: int, place_dim: int, params: dict,
                 seed: int = 0) -> None:
        self.pop = LifPopulation(n_neurons, params)
        self.place_dim = place_dim
        rng = np.random.default_rng(seed)

        n_conn = int(0.05 * place_dim * n_neurons)
        pre = rng.integers(0, place_dim, n_conn)
        post = rng.integers(0, n_neurons, n_conn)
        self.W = sparse.csr_matrix(
            (rng.uniform(0.2, 0.8, n_conn), (post, pre)),
            shape=(n_neurons, place_dim),
        )
        self.gate_open = False   # set True while punishment is present
        self._pattern = self.W.copy()
        self._pattern.data = np.ones_like(self._pattern.data)

    def step(self, place_spikes: np.ndarray) -> np.ndarray:
        self.pop.step(self.W @ place_spikes)
        s_amy = self.pop.last_step_spikes()

        if self.gate_open and place_spikes.any() and s_amy.any():
            # three-factor rule: pre AND post AND dopamine gate
            gate = self.pop.current_dopamine() < 0.05
            if gate:
                ltp = sparse.diags(s_amy.astype(float)) @ self._pattern @ \
                      sparse.diags(place_spikes.astype(float))
                W = self.W + 0.01 * ltp
                np.clip(W.data, 0.0, 5.0, out=W.data)
                self.W = W
        return s_amy

    def reset(self) -> None:
        self.pop.reset()
