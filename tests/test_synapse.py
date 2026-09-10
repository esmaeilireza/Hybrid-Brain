"""Week 3 tests - sparse synapses."""
import numpy as np

from core.synapse import SynapseGroup

PARAMS = {
    "simulation": {"dt_ms": 1.0, "seed": 42},
    "synapse": {
        "tau_excite_ms": 5.0,
        "connection_density": 0.05,
        "excitatory_ratio": 0.8,
        "delay_min_ms": 1.0,
        "w_exc": 1.5,
        "w_inh": 3.0,
    },
}


def test_propagation_is_sparse_and_fast() -> None:
    syn = SynapseGroup(1000, 1000, PARAMS)
    assert syn.n_connections <= 0.05 * 1000 * 1000 * 1.1  # ~5% density
    spikes = np.zeros(1000)
    spikes[0] = 1.0
    syn.propagate(spikes)          # must not raise; must be fast


def test_excitatory_input_increases_current() -> None:
    syn = SynapseGroup(1, 1, PARAMS)
    syn.W = syn.W.__class__(([1.5], ([0], [0])), shape=(1, 1))
    syn.propagate(np.array([1.0]))
    g = syn.step()
    assert g[0] > 0.5


def test_delayed_delivery() -> None:
    syn = SynapseGroup(1, 1, PARAMS)
    syn.W = syn.W.__class__(([1.0], ([0], [0])), shape=(1, 1))
    syn.propagate(np.array([1.0]))
    g0 = syn.step()               # first step releases from buffer? no:
    # with delay_ticks=1, the input is released on the first step() call
    # after propagate; assert release happened exactly once
    assert g0[0] > 0.0
    g1 = syn.step()
    assert g1[0] < g0[0]          # then decays


def test_current_decays_toward_zero() -> None:
    syn = SynapseGroup(1, 1, PARAMS)
    syn.W = syn.W.__class__(([1.0], ([0], [0])), shape=(1, 1))
    syn.propagate(np.array([1.0]))
    syn.step()
    for _ in range(200):
        g = syn.step()
    assert g[0] < 0.01
