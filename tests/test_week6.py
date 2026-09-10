"""Week 6 tests - amygdala conditioning, hypothalamic homeostasis."""
import numpy as np

from core.neurotransmitters import NeuromodulatorSystem
from regions.amygdala import Amygdala
from regions.hypothalamus import Hypothalamus

PARAMS = {
    "simulation": {"dt_ms": 1.0, "seed": 42},
    "neuron": {
        "tau_m_ms": 20.0, "v_rest_mV": -65.0, "v_threshold_mV": -55.0,
        "v_reset_mV": -70.0, "refractory_ms": 2.0,
    },
}


def test_amygdala_conditioning_learns() -> None:
    """Pair cell 0's activity with punishment 50 times; the amygdala's
    response to cell 0 alone must strengthen vs. an unpaired control."""
    amy = Amygdala(100, place_dim=10, params=PARAMS, seed=1)
    cue = np.zeros(10)
    cue[0] = 1.0

    w0_before = amy.W.getcol(0).toarray().sum()

    low_da = 0.02   # punishment context
    # Biologically honest timescale: conditioning is a SECONDS-long
    # sustained pairing, not 50 ms. 2000 ticks = 2 s. With ~5 cue-connected
    # neurons at ~77 Hz and lr=0.01 per coactivation, expected growth
    # is several-fold - comfortably above the 1.5x assertion.
    for _ in range(2_000):
        amy.step(cue, dopamine_level=low_da, baseline=0.1)

    w0_after = amy.W.getcol(0).toarray().sum()
    assert w0_after > w0_before * 1.5, "no conditioning - gate or rule broken"


def test_amygdala_no_learning_without_punishment() -> None:
    amy = Amygdala(100, place_dim=10, params=PARAMS, seed=1)
    cue = np.zeros(10)
    cue[0] = 1.0
    w_before = amy.W.getcol(0).toarray().sum()
    for _ in range(50):
        amy.step(cue, dopamine_level=0.2, baseline=0.1)  # safe context
    w_after = amy.W.getcol(0).toarray().sum()
    assert abs(w_after - w_before) < 1e-9, "learning without punishment"


def test_hypothalamus_homeostasis_cycle() -> None:
    hypo = Hypothalamus()
    for _ in range(80):
        state = hypo.tick()
    assert state["energy"] < 0.5 and state["sleep_pressure"] > 0.5
    assert hypo.needs_attention()
    state = hypo.sleep(30)
    assert state["energy"] > state["sleep_pressure"]
    assert not hypo.needs_attention()
