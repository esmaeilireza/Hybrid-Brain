"""Spike-Timing-Dependent Plasticity - fully sparse, fully vectorized.

Conventions (documented, not accidental):
  - Traces decay exponentially; updates use the traces from BEFORE the
    current tick's spikes, then the current spikes are added to traces.
  - Potentiation: on a postsynaptic spike, strengthen synapses from
    presynaptic neurons whose trace is high (pre-before-post causality).
  - Depression: on a presynaptic spike, weaken synapses onto postsynaptic
    neurons whose trace is high (post-before-pre causality).
  - Both updates are sparse diagonal-pattern products: O(nnz), correct
    for any (n_pre, n_post) population sizes.
"""
from __future__ import annotations

import numpy as np
from scipy import sparse

from core.synapse import SynapseGroup


class STDP:
    """STDP rule attached to one SynapseGroup."""

    def __init__(self, synapse: SynapseGroup, params: dict) -> None:
        st = params["stdp"]
        self.syn = synapse
        self.dt = float(params["simulation"]["dt_ms"])
        self.a_plus = float(st["a_plus"])
        self.a_minus = float(st["a_minus"])
        self.tau_plus = float(st["tau_plus_ms"])
        self.tau_minus = float(st["tau_minus_ms"])
        self.w_max = float(st.get("w_max", 5.0))

        n_pre, n_post = synapse.n_pre, synapse.n_post
        self.pre_trace = np.zeros(n_pre)
        self.post_trace = np.zeros(n_post)
        self._decay_pre = 1.0 - self.dt / self.tau_plus
        self._decay_post = 1.0 - self.dt / self.tau_minus

        # Fixed binary connectivity pattern (weights change, wiring does not)
        self._pattern = synapse.W.copy()
        self._pattern.data = np.ones_like(self._pattern.data)

    def step(self, s_pre: np.ndarray, s_post: np.ndarray) -> None:
        """One tick of STDP given boolean spike masks (pre and post)."""
        W = self.syn.W
        s_pre_f = s_pre.astype(float)
        s_post_f = s_post.astype(float)

        # Depression: pre spikes x post trace -> weaken those synapses.
        # diag(post_trace) @ Pattern @ diag(s_pre) has shape (n_post, n_pre).
        if s_pre.any():
            W = W - self.a_minus * (
                sparse.diags(self.post_trace) @ self._pattern @ sparse.diags(s_pre_f)
            )

        # Potentiation: post spikes x pre trace -> strengthen those synapses.
        if s_post.any():
            W = W + self.a_plus * (
                sparse.diags(s_post_f) @ self._pattern @ sparse.diags(self.pre_trace)
            )

        np.clip(W.data, -self.w_max, self.w_max, out=W.data)
        self.syn.W = W

        # Trace update AFTER applying this tick's updates
        self.pre_trace = self.pre_trace * self._decay_pre + s_pre_f
        self.post_trace = self.post_trace * self._decay_post + s_post_f
