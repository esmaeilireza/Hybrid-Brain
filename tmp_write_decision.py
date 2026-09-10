from pathlib import Path

decision_py = '''"""Decision making - Kahneman System 1/2 with a familiarity router.

Design (Month 4, honestly simplified):
  - System 1 (fast): direct ValueTable lookup when the state is FAMILIAR
    (seen before, with differentiated values). No deliberation.
  - System 2 (slow): one-step lookahead utility computation for NOVEL
    states or high-stakes choices. For each action, the utility of the
    NEXT position's best value is computed (bootstrap-style lookahead),
    weighted by loss aversion (losses count ~2x gains).
  - Router: familiarity = visit count of the position. Familiar ->
    System 1; novel (0 visits) -> System 2. This is a lookup, not a
    philosophy.

Biases (roadmap realism requirements):
  - Loss aversion (Kahneman-Tversky, lambda simplified to 2.0): in
    System 2 utility, negative outcome deltas are weighted 2x.
  - Anchoring: first-encounter estimates are sticky - implemented as a
    reduced learning rate while visit_count < anchor_visits, decaying
    to the normal rate. (The first value seen anchors the estimate.)
"""
from __future__ import annotations

import numpy as np


class DecisionMaker:
    def __init__(self, value_table, visits: dict | None = None,
                 familiarity_threshold: int = 1,
                 anchor_visits: int = 5,
                 loss_aversion: float = 2.0,
                 gamma: float = 0.9) -> None:
        self.values = value_table
        self.visits = visits if visits is not None else {}
        self.familiarity_threshold = familiarity_threshold
        self.anchor_visits = anchor_visits
        self.loss_aversion = loss_aversion
        self.gamma = gamma
        self.last_system = 1          # which system decided last
        self._n_actions = 4

    # ---- router ----
    def route(self, position: tuple[int, int]) -> int:
        """1 = System 1 (familiar), 2 = System 2 (novel)."""
        visits_here = self.visits.get(position, 0)
        if visits_here >= self.familiarity_threshold:
            return 1
        return 2

    # ---- System 1: direct lookup ----
    def system1_action(self, position: tuple[int, int]) -> int:
        v = self.values.values_at(position)
        return int(np.argmax(v))

    # ---- System 2: one-step lookahead utility with loss aversion ----
    def system2_action(self, position: tuple[int, int],
                       neighbor_values: dict[tuple[int, int], np.ndarray]
                       ) -> int:
        """neighbor_values: {next_position: value_vector of that
        position}. The caller supplies the actual neighbors (grid
        topology belongs to the world, not the decision maker)."""
        best_action, best_utility = 0, -np.inf
        for action in range(self._n_actions):
            # each action maps to one neighbor by grid convention
            # (up, down, left, right) - caller provides aligned dict
            nxt = neighbor_values.get(action)
            if nxt is None:
                utility = -10.0   # unknown neighbor: assume worst
            else:
                # outcome delta relative to current state value
                current_best = float(self.values.values_at(position).max())
                outcome = float(nxt.max())
                delta = outcome - current_best
                if delta < 0:
                    delta *= self.loss_aversion   # losses weigh 2x
                utility = self.gamma * outcome + delta
            if utility > best_utility:
                best_utility, best_action = utility, action
        return best_action

    # ---- anchored learning rate for the reinforcement loop ----
    def learning_rate_for(self, position: tuple[int, int]) -> float:
        """Anchor effect: reduced lr while the state is new, decaying
        to normal once past anchor_visits."""
        v = self.visits.get(position, 0)
        if v >= self.anchor_visits:
            return 1.0
        # linear ramp 0.4 -> 1.0 across the anchor window
        return 0.4 + 0.6 * (v / max(self.anchor_visits, 1))

    # ---- top-level decision ----
    def decide(self, position: tuple[int, int],
               neighbor_values: dict[tuple[int, int], np.ndarray]
               ) -> tuple[int, int]:
        """Returns (action, system_used)."""
        system = self.route(position)
        if system == 1:
            action = self.system1_action(position)
        else:
            action = self.system2_action(position, neighbor_values)
        self.last_system = system
        return action, system
'''

test_py = '''"""Week 14 - System 1/2 routing, loss aversion, anchoring."""
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
'''

Path("cognition/decision_making.py").write_text(decision_py, encoding="utf-8")
Path("tests/test_decision.py").write_text(test_py, encoding="utf-8")
print("decision_making.py + test_decision.py written")