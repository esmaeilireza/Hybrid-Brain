"""Week 6 part 2 - PFC working memory, BG action selection."""
import numpy as np

from regions.basal_ganglia import BasalGanglia
from regions.prefrontal import PrefrontalCortex

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
}


def test_pfc_persists_through_delay() -> None:
    """Cue 50 ticks, then 200 silent ticks. Memory = E activity stays
    above baseline through the delay, then is bounded (no runaway)."""
    pfc = PrefrontalCortex(200, PARAMS, seed=1)
    cue = np.full(200, 25.0)
    for _ in range(50):
        pfc.step(cue)
    delay_rates = []
    for _ in range(200):
        s = pfc.step(None)
        delay_rates.append(float(s.sum()))
    mean_delay = np.mean(delay_rates) / 200.0 * 1000.0   # Hz
    assert mean_delay > 2.0, f"memory lost in delay: {mean_delay:.1f} Hz"
    assert mean_delay < 120.0, f"runaway during delay: {mean_delay:.1f} Hz"


def test_bg_selects_strongest_value() -> None:
    bg = BasalGanglia(4, PARAMS, seed=1)
    wins = {1: 0, 2: 0}
    for _ in range(40):
        bg.pop.reset()
        winner = bg.decide(np.array([0.1, 0.9, 0.3, 0.1]))
        wins[winner] = wins.get(winner, 0) + 1
    assert wins[1] > wins[2] * 2, f"selection not following value: {wins}"
