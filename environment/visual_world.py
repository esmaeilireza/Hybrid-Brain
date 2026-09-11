# VisualWorld - PyVista headless 3D scene (World protocol, ADR-017).
# Floor + 3 colored objects. The camera IS the agent: pos + yaw.
# Actions: 0 forward, 1 back, 2 rotate-left, 3 rotate-right.
# Contract: reward ONLY via RewardSensor, action ONLY via the
# actuator, observation ONLY via VisualSensor (visual_cortex).
# Performance: 640x480 offscreen, no shadows (GL verified on 960M).
from __future__ import annotations

import numpy as np

import pyvista as pv

ACTIONS = {0: "forward", 1: "back", 2: "left", 3: "right"}
OBJECTS = {
    "red_cube": (3.0, 1.0, "red"),
    "blue_sphere": (5.0, 3.0, "blue"),
    "green_cylinder": (1.0, 4.0, "green"),
}


class VisualSensor:
    def __init__(self, world):
        self._world = world

    def read(self):
        return self._world.render().ravel().astype(float)

    def target_region(self):
        return "visual_cortex"


class RewardSensor:
    def __init__(self, world):
        self._world = world

    def read(self):
        return np.array([self._world.drain_reward()], dtype=float)

    def target_region(self):
        return "reward_pathway"


class MotorActuator:
    def __init__(self, world):
        self._world = world

    def execute(self, action_vector):
        self._world.pending_action = int(np.argmax(action_vector)) % 4


class VisualWorld:
    def __init__(self, params=None, seed: int = 0) -> None:
        env = (params or {}).get("environment", {})
        self.max_steps = int(env.get("max_steps", 100))
        self.step_penalty = float(env.get("step_penalty", -0.01))
        self.goal_object = env.get("goal_object", "red_cube")
        self.goal_reward = float(env.get("goal_reward", 1.0))
        self._reward_buffer = 0.0
        self._steps = 0
        self.pending_action = 0
        self.pos = np.array([1.0, 1.0])
        self.yaw = 0.0
        self._sensor_v = VisualSensor(self)
        self._sensor_r = RewardSensor(self)
        self._motor = MotorActuator(self)
        self._plotter = None

    def render(self):
        # lazy offscreen render; never blocks the sim loop
        if self._plotter is None:
            self._plotter = pv.Plotter(off_screen=True,
                                       window_size=(640, 480))
            self._plotter.add_mesh(
                pv.Plane(i_size=8, j_size=8), color="gray")
            for name, (x, y, color) in OBJECTS.items():
                shape = (pv.Cube() if "cube" in name
                         else pv.Sphere() if "sphere" in name
                         else pv.Cylinder())
                shape.translate((x, y, 0.5), inplace=True)
                self._plotter.add_mesh(shape, color=color)
        self._plotter.camera.position = (self.pos[0], self.pos[1], 1.5)
        self._plotter.camera.focal_point = (
            self.pos[0] + np.cos(self.yaw),
            self.pos[1] + np.sin(self.yaw), 0.5)
        self._plotter.render()   # force fresh raster after camera move
        img = self._plotter.screenshot(return_img=True)
        return img[:, :, :3]

    def reset(self) -> None:
        self.pos = np.array([1.0, 1.0])
        self.yaw = 0.0
        self._reward_buffer = 0.0
        self._steps = 0
        self.pending_action = 0

    def step(self, dt_ms: int) -> None:
        if self.is_episode_done():
            return
        if self.pending_action == 0:
            self.pos = self.pos + np.array(
                [np.cos(self.yaw), np.sin(self.yaw)]) * 0.5
        elif self.pending_action == 1:
            self.pos = self.pos - np.array(
                [np.cos(self.yaw), np.sin(self.yaw)]) * 0.5
        elif self.pending_action == 2:
            self.yaw += np.pi / 4
        else:
            self.yaw -= np.pi / 4
        self.pos = np.clip(self.pos, 0.0, 8.0)
        self._steps += 1
        self._reward_buffer += self.step_penalty
        name, (x, y, _) = next(iter(OBJECTS.items()))
        if self.goal_object == name:
            if np.linalg.norm(self.pos - np.array([x, y])) < 0.6:
                self._reward_buffer += self.goal_reward

    def sensors(self):
        return [self._sensor_v, self._sensor_r]

    def actuators(self):
        return [self._motor]

    def is_episode_done(self) -> bool:
        return self._steps >= self.max_steps

    def drain_reward(self) -> float:
        r = self._reward_buffer
        self._reward_buffer = 0.0
        return r
