# Cortex6 v3 - rate-coded ff with ROW-SUM NORMALIZATION (ADR-004
# homeostatic precedent): every L5-class neuron receives the same
# total in-strength (w_ff x 8), independent of density/pre-count.
# drive = in_strength x prev_rate. Persistent EMA rate (memory).
from __future__ import annotations

import numpy as np

from core.neuron import LifPopulation
from core.synapse import SynapseGroup

LAYER_SIZES = (4, 2, 2, 4, 3, 2)


class _Layer:
    def __init__(self, n_exc, params, rng):
        n_inh = max(1, n_exc // 4)
        self.pop_e = LifPopulation(n_exc, params)
        self.pop_i = LifPopulation(n_inh, params)
        mk = lambda x, y, pol: SynapseGroup(
            x, y, params, rng, polarity=pol)
        self.e2e = mk(n_exc, n_exc, "excitatory")
        self.e2i = mk(n_exc, n_inh, "excitatory")
        self.i2e = mk(n_inh, n_exc, "inhibitory")
        self.n_exc = n_exc

    def step(self, drive):
        s_e = self.pop_e.step(drive + self.e2e.g + self.i2e.g)
        self.pop_i.step(self.e2i.g)
        f_e = s_e.astype(float)
        f_i = self.pop_i.last_spikes.astype(float)
        self.e2e.propagate(f_e)
        self.e2i.propagate(f_e)
        self.i2e.propagate(f_i)
        self.e2e.step()
        self.e2i.step()
        self.i2e.step()
        return s_e


class Cortex6:
    def __init__(self, n_total, params, layer_density=0.10,
                 w_ff=45.0, seed: int = 0) -> None:
        unit = max(1, n_total // sum(LAYER_SIZES))
        rng = np.random.default_rng(seed)
        self.layers = [_Layer(s * unit, params, rng)
                       for s in LAYER_SIZES]
        self.n_in = self.layers[0].n_exc
        self._rate = [np.zeros(l.n_exc) for l in self.layers]
        self.ff = []
        in_strength = w_ff * 8.0   # target incoming sum per neuron
        for k in range(5):
            n_pre = self.layers[k].n_exc
            n_post = self.layers[k + 1].n_exc
            n_conn = int(layer_density * n_pre * n_post)
            rows = rng.integers(0, n_post, n_conn)
            cols = rng.integers(0, n_pre, n_conn)
            w = np.full(n_conn, 1.0)
            m = np.zeros((n_post, n_pre))
            m[rows, cols] = w
            # homeostatic row-sum normalization (ADR-004 pattern):
            rs = m.sum(axis=1)
            scale = np.where(rs > 0, in_strength / np.maximum(rs, 1e-9), 0.0)
            m *= scale[:, None]
            from scipy import sparse
            self.ff.append(sparse.csr_matrix(m))

    def step(self, input_drive=None):
        drive = np.zeros(self.n_in)
        if input_drive is not None:
            src = np.asarray(input_drive, dtype=float)
            n = min(len(src), self.n_in)
            drive[:n] = src[:n]
        counts = []
        for k, layer in enumerate(self.layers):
            if k == 0:
                s = layer.step(drive)
            else:
                s = layer.step(self.ff[k - 1] @ self._rate[k - 1])
            self._rate[k] = 0.9 * self._rate[k] + 0.1 * s.astype(
                float)
            counts.append(int(s.sum()))
        return counts
