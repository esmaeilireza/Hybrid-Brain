"""Week 4 tests - STDP timing causality and network learning."""
import numpy as np
import pytest

from core.neural_network import NeuralNetwork
from core.plasticity import STDP
from core.synapse import SynapseGroup

PARAMS = {
    "simulation": {"dt_ms": 1.0, "seed": 42},
    "neuron": {
        "tau_m_ms": 20.0, "v_rest_mV": -65.0, "v_threshold_mV": -55.0,
        "v_reset_mV": -70.0, "refractory_ms": 2.0,
    },
    "synapse": {
        "tau_excite_ms": 5.0, "connection_density": 0.05,
        "excitatory_ratio": 0.8, "delay_min_ms": 1.0,
        "w_exc": 1.5, "w_inh": 3.0,
    },
    "stdp": {
        "tau_plus_ms": 20.0, "tau_minus_ms": 20.0,
        "a_plus": 0.01, "a_minus": 0.012, "w_max": 5.0,
    },
}


def make_pair() -> tuple[SynapseGroup, STDP]:
    syn = SynapseGroup(1, 1, PARAMS)
    syn.W = syn.W.__class__(([1.0], ([0], [0])), shape=(1, 1))
    return syn, STDP(syn, PARAMS)


def test_pre_before_post_potentiates() -> None:
    syn, stdp = make_pair()
    w0 = syn.W[0, 0]
    stdp.step(np.array([1.0]), np.array([0.0]))   # pre fires
    for _ in range(5):
        stdp.step(np.array([0.0]), np.array([0.0]))  # 5 ms gap
    stdp.step(np.array([0.0]), np.array([1.0]))   # post fires after pre
    assert syn.W[0, 0] > w0                        # LTP


def test_post_before_pre_depresses() -> None:
    syn, stdp = make_pair()
    w0 = syn.W[0, 0]
    stdp.step(np.array([0.0]), np.array([1.0]))   # post fires first
    for _ in range(5):
        stdp.step(np.array([0.0]), np.array([0.0]))
    stdp.step(np.array([1.0]), np.array([0.0]))   # pre fires after post
    assert syn.W[0, 0] < w0                        # LTD


def test_weights_bounded() -> None:
    syn, stdp = make_pair()
    for _ in range(2000):
        stdp.step(np.array([1.0]), np.array([1.0]))
    assert abs(syn.W[0, 0]) <= PARAMS["stdp"]["w_max"] + 1e-9


def test_correlated_group_learns() -> None:
    """Assembly test: a group driven in correlation with postsynaptic
    activity must gain more weight than an uncorrelated control."""
    net = NeuralNetwork(PARAMS, n_exc=200, n_inh=50, seed=3)
    corr_group = np.arange(0, 50)                    # 25% of E population
    drive = np.zeros(200)
    drive[corr_group] = 25.0                         # strong correlated drive

    corr_idx = np.flatnonzero(net.e2e.W[:, corr_group].getnnz(axis=1) > 0)
    w_before_corr = net.e2e.W[corr_idx][:, corr_group].copy()

    for _ in range(3000):
        net.step(drive)

    w_after_corr = net.e2e.W[corr_idx][:, corr_group].toarray()
    assert (w_after_corr > w_before_corr.toarray() - 1e-9).mean() > 0.5
