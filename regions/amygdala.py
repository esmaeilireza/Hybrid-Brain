"""Amygdala - threat detection via dopamine-gated plasticity.

Corrected contract (design review): neuromodulation is an INPUT to
regions, not a property of neuron populations. The gate level arrives
as an argument - dopamine < baseline means punishment context.

Conditioning protocol:
  - With gate open (punishment present): coactive place-cell -> amygdala
    synapses potentiate (three-factor rule: pre AND post AND dopamine).
  - Result: after pairing region R with punishment, the amygdala fires
    at R's place cells WITHOUT punishment - learned fear (Pavlovian)."""
from __future__ import annotations

import numpy as np
from scipy import sparse

from core.neuron import LifPopulation


class Amygdala:
    def __init__(self, n_neurons: int, place_dim: int, params: dict,
                 lr: float = 0.01, w_max: float = 5.0, gain: float = 40.0,
                 seed: int = 0) -> None:
        self.pop = LifPopulation(n_neurons, params)
        self.gain = gain
        self.lr = lr
        self.w_max = w_max
        rng = np.random.default_rng(seed)

        n_conn = int(0.05 * place_dim * n_neurons)
        pre = rng.integers(0, place_dim, n_conn)
        post = rng.integers(0, n_neurons, n_conn)
        self.W = sparse.csr_matrix(
            (rng.uniform(0.2, 0.8, n_conn), (post, pre)),
            shape=(n_neurons, place_dim),
        )
        self._pattern = self.W.copy()
        self._pattern.data = np.ones_like(self._pattern.data)
        self.last_spikes = np.zeros(n_neurons, dtype=bool)

    def step(self, place_spikes: np.ndarray, dopamine_level: float,
             baseline: float = 0.1) -> np.ndarray:
        # gain: calibrated so a single ~0.5mV afferent drives ~20mV
        # effective current -> above threshold (Week-2 f-I curve)
        self.last_spikes = self.pop.step(self.gain * (self.W @ place_spikes))

        punishment = dopamine_level < baseline
        if punishment and place_spikes.any() and self.last_spikes.any():
            ltp = (sparse.diags(self.last_spikes.astype(float))
                   @ self._pattern
                   @ sparse.diags(place_spikes.astype(float)))
            W = self.W + self.lr * ltp
            np.clip(W.data, 0.0, self.w_max, out=W.data)
            self.W = W
        return self.last_spikes

    def reset(self) -> None:
        self.pop.reset()
        self.last_spikes[:] = False
