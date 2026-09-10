"""Week 18 - somatic markers: accrual, extinction, decision bias."""
import numpy as np

from emotions.somatic import SomaticMarkerMap


class FakeValues:
    """Minimal ValueTable stand-in with controlled action values."""
    def __init__(self, table):
        self.t = table
    def values_at(self, pos):
        return np.array(self.t[pos], dtype=float)


def make_dm(values, visits, somatic=None):
    from cognition.decision_making import DecisionMaker
    return DecisionMaker(values, visits=visits, somatic=somatic)


def test_penalty_accrues_bounded_on_punishment() -> None:
    m = SomaticMarkerMap()
    for _ in range(20):
        m.record((3, 3), punishment=-0.8, arousal=0.9)
    assert m.penalty((3, 3)) == -0.5          # bounded at max_penalty


def test_no_marker_from_reward() -> None:
    m = SomaticMarkerMap()
    m.record((1, 1), punishment=+0.9, arousal=0.9)
    assert m.penalty((1, 1)) == 0.0
    m.record((1, 1), punishment=-0.5, arousal=0.0)   # arousal gate
    assert m.penalty((1, 1)) == 0.0


def test_extinction_decays_marker() -> None:
    m = SomaticMarkerMap()
    m.record((2, 2), punishment=-0.8, arousal=1.0)
    p0 = m.penalty((2, 2))
    for _ in range(10):
        m.note_safe_visit((2, 2))
    assert m.penalty((2, 2)) > p0
    assert m.penalty((2, 2)) == 0.0           # fully extinct


def test_system1_avoids_marked_neighbor() -> None:
    # near-tied values: action 0 slightly better on the table alone
    values = FakeValues({(0, 0): [0.50, 0.49, 0.1, 0.1]})
    dm = make_dm(values, visits={(0, 0): 10})
    assert dm.system1_action((0, 0)) == 0     # no marker -> action 0

    m = SomaticMarkerMap()
    m.record((0, 1), punishment=-0.8, arousal=0.9)   # action 0 leads here
    dm2 = make_dm(values, visits={(0, 0): 10}, somatic=m)
    action = dm2.system1_action((0, 0),
                                neighbor_positions={0: (0, 1), 1: (1, 0)})
    assert action == 1                        # bias flips the choice


def test_system2_avoids_marked_neighbor() -> None:
    values = FakeValues({(0, 0): [0.5, 0.5, 0.5, 0.5],
                         (0, 1): [0.9, 0.9, 0.9, 0.9],
                         (1, 0): [0.7, 0.7, 0.7, 0.7]})
    dm = make_dm(values, visits={(0, 0): 0})  # novel -> System 2
    nb_vals = {0: values.values_at((0, 1)), 1: values.values_at((1, 0))}
    assert dm.system2_action((0, 0), nb_vals) == 0    # (0,1) better alone

    m = SomaticMarkerMap()
    m.record((0, 1), punishment=-0.8, arousal=1.0)
    dm2 = make_dm(values, visits={(0, 0): 0}, somatic=m)
    action = dm2.system2_action((0, 0), nb_vals,
                                neighbor_positions={0: (0, 1), 1: (1, 0)})
    assert action == 1


def test_no_somatic_backward_compatible() -> None:
    values = FakeValues({(0, 0): [0.5, 0.2, 0.1, 0.1]})
    dm = make_dm(values, visits={(0, 0): 3})
    nb_vals = {a: values.values_at((0, 0)) for a in range(4)}
    action, system = dm.decide((0, 0), nb_vals)
    assert (action, system) == (0, 1)         # unchanged Month 4 behavior
