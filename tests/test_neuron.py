"""Week 2 tests - physical correctness, not just "no crash"."""
import numpy as np
import pytest

from core.neuron import LifPopulation


PARAMS = {
    "simulation": {"dt_ms": 1.0},
    "neuron": {
        "tau_m_ms": 20.0,
        "v_rest_mV": -65.0,
        "v_threshold_mV": -55.0,
        "v_reset_mV": -70.0,
        "refractory_ms": 2.0,
    },
}


def make_pop(n: int = 4) -> LifPopulation:
    return LifPopulation(n, PARAMS)


def test_resting_stays_resting() -> None:
    pop = make_pop()
    for _ in range(200):
        assert not pop.step(np.zeros(pop.n)).any()


def test_spikes_above_rheobase() -> None:
    pop = make_pop()
    weak_spikes = sum(pop.step(np.full(pop.n, 5.0)).any() for _ in range(500))

    pop.reset()
    strong_spikes = sum(pop.step(np.full(pop.n, 20.0)).sum() for _ in range(500))

    assert weak_spikes == 0
    assert strong_spikes > 5


def test_refractory_respected() -> None:
    pop = make_pop(1)
    spike_ticks = [t for t in range(500) if pop.step(np.full(1, 30.0))[0]]
    assert len(spike_ticks) > 10
    gaps = np.diff(spike_ticks)
    assert (gaps >= 3).all()


def test_subtick_interpolation_precision() -> None:
    """Heart of ADR-003: spike_frac in [0, 1), never NaN after a spike."""
    pop = make_pop(1)
    for _ in range(200):
        if pop.step(np.full(1, 25.0))[0]:
            frac = pop.last_spike_frac[0]
            assert not np.isnan(frac)
            assert 0.0 <= frac < 1.0
            return
    pytest.fail("neuron never spiked in 200 ticks - dynamics broken")


def test_firing_rate_monotonic_in_current() -> None:
    rates = []
    for current in (15.0, 25.0, 40.0):
        pop = make_pop(1)
        rates.append(sum(pop.step(np.full(1, current)).sum() for _ in range(1_000)))
    assert rates[0] < rates[1] < rates[2]
