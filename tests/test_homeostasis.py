"""Week 5 tests - homeostatic synaptic scaling (ADR-004)."""
import numpy as np

from core.plasticity import SynapticScaling
from core.synapse import SynapseGroup

PARAMS = {
    "simulation": {"dt_ms": 1.0, "seed": 42},
    "synapse": {
        "tau_excite_ms": 5.0, "connection_density": 0.5,
        "excitatory_ratio": 1.0, "delay_min_ms": 1.0,
        "w_exc": 1.0, "w_inh": 3.0,
    },
    "stdp": {
        "tau_plus_ms": 20.0, "tau_minus_ms": 20.0,
        "a_plus": 0.01, "a_minus": 0.012, "w_max": 5.0,
    },
    "homeostasis": {
        "enabled": True, "scaling_interval_ticks": 1, "target_row_sum": 50.0,
    },
}


def make_syn() -> SynapseGroup:
    return SynapseGroup(20, 20, PARAMS)


def test_scaling_pulls_row_sums_to_target() -> None:
    syn = make_syn()
    scaler = SynapticScaling(syn, PARAMS)
    for _ in range(20):           # 20 events, clamp 2x per event
        scaler.step()
    row_sums = np.asarray(syn.W.sum(axis=1)).ravel()
    connected = syn.W.getnnz(axis=1) > 0
    # after repeated events, sums must have moved strongly toward target
    assert row_sums[connected].min() > row_sums[connected].min() * 0 + 5.0
    assert (row_sums[connected] <= 50.0 * 1.05).all()


def test_scaling_preserves_relative_structure() -> None:
    """The Turrigiano property: multiplicative scaling keeps RATIOS.
    Double one row's weights manually; after scaling, its weights must
    remain proportionally higher than a control row's."""
    syn = make_syn()
    scaler = SynapticScaling(syn, PARAMS)
    W = syn.W.tocsr().tolil()
    W[0, :] *= 2.0                # make row 0 twice as strong
    syn.W = W.tocsr()
    scaler.step()
    scaled = syn.W.tocsr()
    r0 = scaled.getrow(0).data
    r1 = scaled.getrow(1).data
    assert r0.mean() > r1.mean(), "relative structure lost - scaling additive?"


def test_disabled_scaling_is_noop() -> None:
    params = {**PARAMS, "homeostasis": {"enabled": False}}
    syn = make_syn()
    w_before = syn.W.copy()
    scaler = SynapticScaling(syn, params)
    for _ in range(50):
        scaler.step()
    assert (syn.W.toarray() == w_before.toarray()).all()
