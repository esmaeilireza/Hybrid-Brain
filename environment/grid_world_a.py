"""GridWorld-A - the first environment. Level A: plain grid, no obstacles.

Contract compliance (core/interfaces.py):
  - reward is NEVER a return value of step(); it accumulates and is read
    through RewardSensor, exactly like the dopamine pathway it will feed
  - the brain acts only through MotorActuator
  - perception flows only through GridSensor
"""
from __future__ import annotations

import numpy as np

from core.interfaces import Actuator, Sensor, World  # noqa: F401 (Protocol check)

ACTIONS = {"up": 0, "down": 1, "left": 2, "right": 3}
_DXY = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}


class GridSensor:
    """Spatial observation: agent position one-hot over the grid."""

    def __init__(self, world: "GridWorldA") -> None:
        self._world = world

    def read(self) -> np.ndarray:
        r, c = self._world.agent
        obs = np.zeros(self._world.size * self._world.size, dtype=np.float64)
        obs[r * self._world.size + c] = 1.0
        return obs

    def target_region(self) -> str:
        return "thalamus_spatial"


class RewardSensor:
    """Reads accumulated reward and drains it - the dopamine pathway feed."""

    def __init__(self, world: "GridWorldA") -> None:
        self._world = world

    def read(self) -> np.ndarray:
        return np.array([self._world.drain_reward()], dtype=np.float64)

    def target_region(self) -> str:
        return "reward_pathway"


class MotorActuator:
    """Receives the brain's action vector; argmax selects the move."""

    def __init__(self, world: "GridWorldA") -> None:
        self._world = world

    def execute(self, action_vector: np.ndarray) -> None:
        action = int(np.argmax(action_vector))
        self._world.pending_action = action % len(ACTIONS)


class GridWorldA:
    """20x20 grid. Agent starts random, goal fixed at far corner.
    Reward: +1.0 on reaching the goal, -0.01 per step (time pressure)."""

    def __init__(self, params: dict | None = None, seed: int = 0) -> None:
        env = (params or {}).get("environment", {})
        self.size = int(env.get("grid_size", 20))
        self.max_steps = int(env.get("max_steps", 300))
        self.step_penalty = float(env.get("step_penalty", -0.01))
        self.rng = np.random.default_rng(seed)

        self.goal = (self.size - 1, self.size - 1)
        self.agent: tuple[int, int] = (0, 0)
        self.pending_action: int = 0
        self._reward_buffer: float = 0.0
        self._steps: int = 0

        self._grid_sensor = GridSensor(self)
        self._reward_sensor = RewardSensor(self)
        self._motor = MotorActuator(self)

    # --- World protocol ---
    def reset(self) -> None:
        self.agent = (
            int(self.rng.integers(0, self.size)),
            int(self.rng.integers(0, self.size)),
        )
        self.pending_action = 0
        self._reward_buffer = 0.0
        self._steps = 0

    def step(self, dt_ms: int) -> None:
        if self.is_episode_done():
            return
        dr, dc = _DXY[self.pending_action]
        r = min(max(self.agent[0] + dr, 0), self.size - 1)
        c = min(max(self.agent[1] + dc, 0), self.size - 1)
        self.agent = (r, c)
        self._steps += 1
        self._reward_buffer += self.step_penalty
        if self.agent == self.goal:
            self._reward_buffer += 1.0

    def sensors(self) -> list[Sensor]:
        return [self._grid_sensor, self._reward_sensor]

    def actuators(self) -> list[Actuator]:
        return [self._motor]

    def is_episode_done(self) -> bool:
        return self.agent == self.goal or self._steps >= self.max_steps

    # --- internal ---
    def drain_reward(self) -> float:
        r = self._reward_buffer
        self._reward_buffer = 0.0
        return r
