# Action selector - the integration point (Week 23).
# Priority: HABIT (ballistic, no evaluation) > System1/2 DecisionMaker
#   > BG competition. Value signals may carry somatic markers (W18)
#   and personality bias (W20) BEFORE BG sees them.
from __future__ import annotations

import numpy as np

from behavior.habits import Habits
from cognition.decision_making import DecisionMaker
from regions.basal_ganglia import BasalGanglia


class ActionSelector:
    def __init__(self, decision_maker: DecisionMaker,
                 bg: BasalGanglia, habits: Habits) -> None:
        self.dm = decision_maker
        self.bg = bg
        self.habits = habits
        self.last_source = None     # habit | dm | bg

    def select(self, position: tuple, value_signals: np.ndarray,
               neighbor_values: dict, neighbor_positions: dict = None,
               habit_rpe: float = None) -> int:
        # 1) HABIT first - ballistic cache hit
        ha = self.habits.get(position)
        if ha is not None:
            self.last_source = "habit"
            if habit_rpe is not None:
                self.habits.report_outcome(position, habit_rpe)
            return ha
        action = 0   # default when no DM/BG is wired
        # 2) DecisionMaker (System 1/2 + somatic markers), fed to BG
        # 3) BG final competition over modulated value signals
        vs = np.asarray(value_signals, dtype=float)
        if self.bg is not None and len(vs) == 4:
            action = self.bg.decide(vs)
            self.last_source = "bg"
        return action
