"""Week 5 tests - thalamus relay and hippocampal place fields.

The thalamus test is deterministic: it reads the afferent projection
and asserts that exactly the matched neurons can fire. No statistical
thresholds - a threshold guessed from wrong arithmetic failed here once
(fired=12 vs asserted >50) and the honest fix was a contract test."""
import numpy as np

from regions.hippocampus import PlaceCellPopulation
from regions.thalamus import Thalamus

PARAMS = {
    "simulation": {"dt_ms": 1.0, "seed": 42},
    "neuron": {
        "tau_m_ms": 20.0, "v_rest_mV": -65.0, "v_threshold_mV": -55.0,
        "v_reset_mV": -70.0, "refractory_ms": 2.0,
    },
}


def test_thalamus_relays_deterministically() -> None:
    th = Thalamus(200, sensor_dim=400, params=PARAMS, gain=250.0, seed=1)
    obs = np.zeros(400)
    obs[7] = 1.0

    # Contract 1: exactly the neurons with unit 7 among their afferents
    # are allowed to fire (sparse convergent relay = selectivity)
    col = th.projection.getcol(7).toarray().ravel()
    matched = set(np.flatnonzero(col > 0).tolist())
    assert 0 < len(matched) < 40, "afferenent statistics off - check k/dim"

    active: set[int] = set()
    total = 0
    for _ in range(100):
        s = th.step(obs)
        total += int(s.sum())
        active.update(np.flatnonzero(s))

    assert active.issubset(matched), "non-matched neurons fired - leak"
    assert active == matched, f"matched neurons silent: {matched - active}"
    # Contract 2: per-afferent current 25mV -> ~63Hz -> each matched
    # neuron fires several times in 100 ticks
    assert total >= len(matched) * 2, "relay too weak for 25mV drive"


def test_place_cells_fire_near_field_center() -> None:
    pc = PlaceCellPopulation(50, grid_size=20, params=PARAMS, seed=1)
    center = tuple(pc.centers[0])
    near_spikes = sum(pc.step(center).sum() for _ in range(50))
    far = ((center[0] + 10) % 20, (center[1] + 10) % 20)
    pc.pop.reset()
    far_spikes = sum(pc.step(far).sum() for _ in range(50))
    assert near_spikes > 5 * far_spikes, "place field selectivity broken"
