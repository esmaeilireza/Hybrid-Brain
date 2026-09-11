"""GridWorld-B - the personality pressure cooker (Week 20, ADR-007).

A + moving obstacles + collectibles + two goals of different value.
Contract identical to A (interfaces.py): reward ONLY via RewardSensor,
action ONLY via MotorActuator. Reuses A's sensor/actuator classes -
they duck-type on world attributes (audited Week 20).

fixed_start config: both OCEAN arms must start identical, so B
supports a fixed spawn (A randomizes - audited)."""
from __future__ import annotations

import numpy as np

from environment.grid_world_a import (
    GridSensor, MotorActuator, RewardSensor, _DXY)


class GridWorldB:
    def __init__(self, params: dict | None = None, seed: int = 0) -> None:
        env = (params or {}).get("environment", {})
        self.size = int(env.get("grid_size", 20))
        self.max_steps = int(env.get("max_steps", 400))
        self.step_penalty = float(env.get("step_penalty", -0.01))
        self.bump_penalty = float(env.get("bump_penalty", -0.15))
        fs = env.get("fixed_start")
        self._fixed_start = tuple(int(v) for v in fs) if fs else None
        self.rng = np.random.default_rng(seed)

        s = self.size
        self.goal_high = (s - 1, s - 1)
        self.goal_low = (s // 2, s - 1)
        self.goal_values = {self.goal_high: 2.0, self.goal_low: 0.5}

        # a vertical wall with two gaps (columns force route choices)
        self.static_obstacles = {
            (r, s // 2) for r in range(s) if abs(r - s // 2) > 2}

        # two patrolling movers (horizontal, bounce at edges/walls)
        self._movers_init = [
            {"pos": [s // 4, 2], "dir": 1},
            {"pos": [3 * s // 4, s - 3], "dir": -1},
        ]
        self.movers = [dict(m) for m in self._movers_init]

        self._collectibles_init = {(3, 3): 0.5, (s - 3, 3): 0.5,
                                   (s // 2, s - 4): 0.5}
        self.collectibles = dict(self._collectibles_init)

        self.agent: tuple[int, int] = (0, 0)
        self.pending_action: int = 0
        self._reward_buffer = 0.0
        self._steps = 0
        self.bumps = 0                      # personality benchmark stat

        self._grid_sensor = GridSensor(self)
        self._reward_sensor = RewardSensor(self)
        self._motor = MotorActuator(self)

    # --- World protocol ---
    def reset(self) -> None:
        self.agent = self._fixed_start or (0, 0)
        self.pending_action = 0
        self._reward_buffer = 0.0
        self._steps = 0
        self.bumps = 0
        self.collectibles = dict(self._collectibles_init)
        self.movers = [dict(m) for m in self._movers_init]

    def _mover_cells(self) -> set:
        return {tuple(m["pos"]) for m in self.movers}

    def _move_movers(self) -> None:
        for m in self.movers:
            r, c = m["pos"]
            c2 = c + m["dir"]
            if (c2 < 1 or c2 > self.size - 2
                    or (r, c2) in self.static_obstacles
                    or (r, c2) == self.agent):
                m["dir"] *= -1
            else:
                m["pos"] = [r, c2]

    def step(self, dt_ms: int) -> None:
        if self.is_episode_done():
            return
        self._move_movers()
        dr, dc = _DXY[self.pending_action]
        r = min(max(self.agent[0] + dr, 0), self.size - 1)
        c = min(max(self.agent[1] + dc, 0), self.size - 1)
        target = (r, c)
        if target in self.static_obstacles or target in self._mover_cells():
            self._reward_buffer += self.bump_penalty
            self.bumps += 1                 # agent stays put
        else:
            self.agent = target
        self._steps += 1
        self._reward_buffer += self.step_penalty
        if target in self.collectibles:
            self._reward_buffer += self.collectibles.pop(target)
        if target in self.goal_values:
            self._reward_buffer += self.goal_values[target]

    def sensors(self) -> list:
        return [self._grid_sensor, self._reward_sensor]

    def actuators(self) -> list:
        return [self._motor]

    def is_episode_done(self) -> bool:
        return (self.agent in self.goal_values
                or self._steps >= self.max_steps)

    def drain_reward(self) -> float:
        r = self._reward_buffer
        self._reward_buffer = 0.0
        return r
