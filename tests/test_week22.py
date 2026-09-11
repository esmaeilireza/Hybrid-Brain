# Week 22 - Cocktail Party with two real streams.
# Acceptance: (1) attended stream gets 3x gain; (2) a loud
# auditory event STEALS attention after dwell (salience > 1.25x);
# (3) auditory salience tracks event on/off.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from cognition.attention import AttentionSystem
from perception.auditory import AuditoryStream

PARAMS = {
    "simulation": {"dt_ms": 1.0},
    "neuron": {"tau_m_ms": 20.0, "v_rest_mV": -65.0,
               "v_threshold_mV": -55.0, "v_reset_mV": -70.0,
               "refractory_ms": 2.0},
}


def test_auditory_salience_tracks_event():
    aud = AuditoryStream(PARAMS, seed=1)
    aud.set_event(True)
    for _ in range(50):
        aud.step()
    on_sal = aud.salience()
    aud.set_event(False)
    for _ in range(50):
        aud.step()
    off_sal = aud.salience()
    assert on_sal > off_sal * 2.0, (on_sal, off_sal)


def test_attended_stream_gets_gain():
    att = AttentionSystem(n_streams=2, seed=0)
    gains = att.step([5.0, 1.0])
    assert gains[0] == 3.0 and gains[1] == 1.0


def test_loud_event_steals_attention():
    att = AttentionSystem(n_streams=2, seed=0)
    att.step([5.0, 1.0])          # visual attended
    for _ in range(4):
        att.step([5.0, 1.0])      # dwell satisfied (min_dwell=3)
    gains = att.step([5.0, 8.0])  # loud auditory: >1.25x salience
    assert att.attended_idx == 1, "auditory failed to steal focus"
    assert gains[1] == 3.0
