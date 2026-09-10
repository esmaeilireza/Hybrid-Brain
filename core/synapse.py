"""Synaptic transmission - sparse, vectorized, delayed.

Architecture rules for this file:
  1. Connectivity is scipy.sparse - biological brains are ~5% dense
  2. Propagation is W @ spikes - a sparse matrix-vector product,
     never a Python loop over neurons
  3. Signed weights: positive = excitatory, negative = inhibitory
  4. Fixed group delay via a ring buffer of pending inputs
"""
from __future__ import annotations

import numpy as np
from scipy import sparse


class SynapseGroup:
    """A sparse synapse group from a pre-population to a post-population.

    Each presynaptic spike injects current into postsynaptic neurons.
    Current decays exponentially (AMPA/GABA-like first-order dynamics).
    """

    def __init__(
        self,
        n_pre: int,
        n_post: int,
        params: dict,
        rng: np.random.Generator | None = None,
    ) -> None:
        self.n_pre = n_pre
        self.n_post = n_post
        self.rng = rng or np.random.default_rng(
            params["simulation"].get("seed", 42)
        )

        s = params["synapse"]
        self.dt = float(params["simulation"]["dt_ms"])
        self.tau_ms = float(s["tau_excite_ms"])  # decay time constant (ms)
        self.density = float(s["connection_density"])
        self.exc_ratio = float(s["excitatory_ratio"])
        self.delay_ticks = int(round(float(s["delay_min_ms"]) / self.dt))
        self.w_exc = float(s.get("w_exc", 1.5))   # excitatory weight (mV-scale)
        self.w_inh = float(s.get("w_inh", 3.0))   # inhibitory magnitude

        # --- Build sparse connectivity: shape (n_post, n_pre) so that
        #     propagation is literally  W @ spikes_pre  (the golden rule) ---
        n_conn = int(self.density * n_pre * n_post)
        pre_idx = self.rng.integers(0, n_pre, size=n_conn)
        post_idx = self.rng.integers(0, n_post, size=n_conn)
        n_exc = int(self.exc_ratio * n_pre)

        weights = np.where(
            pre_idx < n_exc,
            self.rng.uniform(0.7, 1.0, n_conn) * self.w_exc,
            -self.rng.uniform(0.7, 1.0, n_conn) * self.w_inh,
        )
        self.W = sparse.csr_matrix(
            (weights, (post_idx, pre_idx)), shape=(n_post, n_pre)
        )

        # --- Synaptic current state and delay ring buffer ---
        self.g = np.zeros(n_post, dtype=np.float64)
        self._decay = 1.0 - self.dt / self.tau_ms
        if self.delay_ticks <= 0:
            self.delay_ticks = 1
        self._pending: list[np.ndarray] = [
            np.zeros(n_post) for _ in range(self.delay_ticks)
        ]

    def propagate(self, spikes_pre: np.ndarray) -> None:
        """Queue the effect of presynaptic spikes (arrives after delay)."""
        self._pending[-1] += self.W @ spikes_pre

    def step(self) -> np.ndarray:
        """Release due inputs, decay current. Returns current (mV-scale)."""
        released = self._pending.pop(0)
        self._pending.append(np.zeros(self.n_post))
        self.g = self.g * self._decay + released
        return self.g

    @property
    def n_connections(self) -> int:
        return int(self.W.nnz)
