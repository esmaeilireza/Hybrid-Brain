"""Week 3 tests - GridWorld-A contract compliance."""
import numpy as np

from core.interfaces import Actuator, Sensor, World
from environment.grid_world_a import ACTIONS, GridWorldA


def make_world() -> GridWorldA:
    return GridWorldA({"environment": {"grid_size": 20, "max_steps": 300}}, seed=7)


def test_implements_world_protocol() -> None:
    world = make_world()
    assert isinstance(world, World)
    for s in world.sensors():
        assert isinstance(s, Sensor)
    for a in world.actuators():
        assert isinstance(a, Actuator)


def test_agent_moves_within_bounds() -> None:
    world = make_world()
    world.reset()
    world.pending_action = ACTIONS["up"]     # would leave grid from row 0
    world.step(1)
    assert world.agent[0] >= 0               # clamped, not out of bounds


def test_reward_flows_only_through_sensor() -> None:
    world = make_world()
    world.reset()
    world.agent = world.goal                 # teleport to goal
    reward_sensor = world.sensors()[1]
    assert reward_sensor.target_region() == "reward_pathway"


def test_goal_reached_ends_episode_with_reward() -> None:
    world = make_world()
    world.reset()
    world.agent = (world.goal[0], world.goal[1] - 1)  # one step away
    world.pending_action = ACTIONS["right"]
    world.step(1)
    assert world.is_episode_done()
    reward = world.sensors()[1].read()[0]
    assert reward > 0.9                      # +1.0 minus one step penalty


def test_step_penalty_accumulates() -> None:
    world = make_world()
    world.reset()
    world.agent = (0, 0)
    world.pending_action = ACTIONS["down"]
    world.step(1)
    reward = world.sensors()[1].read()[0]
    assert reward < 0.0                      # -0.01 penalty
