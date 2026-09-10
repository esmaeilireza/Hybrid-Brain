"""Week 14 - System 1/2 routing, loss aversion, anchoring."""
import numpy as np

from cognition.consolidation import ValueTable
from cognition.decision_making import DecisionMaker


def make_dm() -> DecisionMaker:
    return DecisionMaker(ValueTable(seed=1), visits={})


def test_router_novel_state_uses_system2() -> None:
    dm = make_dm()
    assert dm.route((9, 9)) == 2, "unvisited state must go to System 2"


def test_router_familiar_state_uses_system1() -> None:
    dm = make_dm()
    dm.visits[(3, 3)] = 5
    dm.values.values_at((3, 3))       # materialize entry
    dm.visits[(3, 3)] += 0            # ensure counted
    assert dm.route((3, 3)) == 1


def test_loss_aversion_changes_symmetric_choice() -> None:
    """A state where action A's neighbor has +value and action B's
    neighbor has equal-magnitude negative delta: with loss aversion,
    the safe action wins; without it, they tie."""
    dm = make_dm()
    dm.loss_aversion = 2.0
    # current position has best value 1.0; action A leads to a neighbor
    # with best 1.5 (delta +0.5), action B to a neighbor with best 0.5
    # (delta -0.5). With lambda=2: A utility ~ 0.9*1.5+0.5, B ~
    # 0.9*0.5-1.0. A must win decisively.
    pos = (5, 5)
    dm.values.reinforce(pos, 0, 1.0)  # give current state a best value
    dm.values.table[pos[0] * 20 + pos[1]] = np.array([1.0, 0.0, 0.0, 0.0])
    nbr_A, nbr_B = (4, 5), (6, 5)
    dm.values.table[nbr_A[0] * 20 + nbr_A[1]] = np.array([1.5, 0.0, 0.0, 0.0])
    dm.values.table[nbr_B[0] * 20 + nbr_B[1]] = np.array([0.5, 0.0, 0.0, 0.0])
    nv = {0: dm.values.values_at(nbr_A), 1: dm.values.values_at(nbr_B)}
    action = dm.system2_action(pos, nv)
    assert action == 0, f"expected safe action A, got {action}"


def test_anchoring_slows_early_learning() -> None:
    dm = make_dm()
    lr_early = dm.learning_rate_for((1, 1))       # 0 visits
    dm.visits[(1, 1)] = 5
    lr_late = dm.learning_rate_for((1, 1))
    assert lr_early < lr_late, "anchor did not slow early learning"
    assert lr_late == 1.0, "late learning should reach full rate"


def test_decide_returns_system_tag() -> None:
    dm = make_dm()
    pos = (2, 2)
    dm.visits[pos] = 3
    dm.values.values_at(pos)
    action, system = dm.decide(pos, {})
    assert system == 1
    pos_novel = (7, 7)
    nv = {a: np.zeros(4) for a in range(4)}
    action, system = dm.decide(pos_novel, nv)
    assert system == 2
