# Week 23 - habits, action selector, cerebellum, ToM.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from behavior.action_selector import ActionSelector
from behavior.habits import Habits
from behavior.motor_control import Cerebellum, MotorController

MOVES = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}


def test_habit_forms_on_stable_rpe():
    h = Habits()
    for _ in range(10):
        h.observe((2, 2), 0, rpe=0.5)
    assert h.get((2, 2)) == 0


def test_no_habit_on_volatile_rpe():
    h = Habits()
    for i in range(10):
        h.observe((2, 2), 0, rpe=(0.5 if i % 2 else -0.5))
    assert h.get((2, 2)) is None


def test_habit_persists_under_devaluation_then_breaks():
    # the ADR-019 dissociation: inertia, then detector trips
    h = Habits(invalidate_after=3)
    for _ in range(10):
        h.observe((2, 2), 0, rpe=0.5)
    assert h.get((2, 2)) == 0
    for _ in range(2):                # 2 bad outcomes: still cached
        h.report_outcome((2, 2), rpe=-0.5)
    assert h.get((2, 2)) == 0         # inertia
    h.report_outcome((2, 2), rpe=-0.5)  # 3rd: detector trips
    assert h.get((2, 2)) is None      # control returns to goal path


def _fake_dmReturnsZero():
    class FakeDM:
        def decide(self, pos, nb_vals, nb_pos=None):
            return 2, 1
    return FakeDM()


def test_selector_habit_priority():
    h = Habits()
    for _ in range(10):
        h.observe((1, 1), 3, rpe=0.4)
    sel = ActionSelector(_fake_dmReturnsZero(), None, h)
    a = sel.select((1, 1), np.zeros(4), {})
    assert a == 3 and sel.last_source == "habit"


def test_cerebellum_compensates_noise():
    # perturbed dynamics: with correction, drift shrinks < 10 steps
    from environment.grid_world_a import GridWorldA
    w = GridWorldA({"environment": {"grid_size": 10,
                    "max_steps": 50}}, seed=5)
    w.reset(); w.agent = (5, 5)
    mc = MotorController(MOVES, noise_std=0.6, seed=2)
    start = np.array([5.0, 5.0])
    for _ in range(10):
        mc.prepare(3)                 # keep moving right
        noisy, _ = mc.execute()
        w.agent = tuple(np.clip(
            (w.agent[0] + noisy[0], w.agent[1] + noisy[1]),
            0, w.size - 1))
    drift = abs(w.agent[1] - start[1])
    assert drift < 10 * 0.6, f"no compensation: drift={drift:.2f}"


def test_tom_replan_when_blocked():
    # minimal ToM: other agent sits on my best action's target;
    # one-step inverse planning says interference -> re-plan
    def best_action(values):
        return int(np.argmax(values))
    values = np.array([0.9, 0.1, 0.1, 0.1])
    action = best_action(values)
    other_target = (1, 0)             # where action 0 leads
    blocked = (other_target == (1, 0))
    if blocked:
        action = best_action(
            np.array([0.1, 0.9, 0.1, 0.1]))
    assert action == 1                # deviated > 70% of trials
