"""Fundamental contract test — Week 1.

This test guarantees that the Sensor/Actuator/World contract is
executable and verifiable from day one.
"""
import numpy as np

from core.interfaces import Actuator, Sensor, World


class _FakeSensor:
    def read(self) -> np.ndarray:
        return np.zeros(10)

    def target_region(self) -> str:
        return "thalamus_visual"


class _FakeActuator:
    def execute(self, action_vector: np.ndarray) -> None:
        assert action_vector.ndim == 1


class _FakeWorld:
    def reset(self) -> None: ...

    def step(self, dt_ms: int) -> None:
        assert dt_ms > 0

    def sensors(self) -> list[Sensor]:
        return [_FakeSensor()]

    def actuators(self) -> list[Actuator]:
        return [_FakeActuator()]

    def is_episode_done(self) -> bool:
        return False


def test_sensor_contract() -> None:
    sensor = _FakeSensor()
    assert isinstance(sensor, Sensor)
    assert sensor.target_region() == "thalamus_visual"


def test_actuator_contract() -> None:
    actuator = _FakeActuator()
    assert isinstance(actuator, Actuator)
    actuator.execute(np.array([1.0, 0.0]))


def test_world_contract() -> None:
    world = _FakeWorld()
    assert isinstance(world, World)
    world.reset()
    world.step(dt_ms=1)
    assert not world.is_episode_done()
    assert len(world.sensors()) == 1
