"""Decision making - Kahneman System 1/2 with a familiarity router.

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
