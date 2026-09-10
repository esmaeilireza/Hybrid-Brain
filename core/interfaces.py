"""
Fundamental Hybrid-Brain contract — ADR-002 compatible, immutable.

Golden rule: the brain never talks directly to the world.
All inputs are Sensors, all outputs are Actuators, all worlds are World.
Raspberry Pi, GridWorld, VisualWorld: they are all just another World.
"""
from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class Sensor(Protocol):
    """Every input to the brain — without exception.

    Implementations: GridWorldSensor, RewardSensor, VisualSensor,
    in the future: Raspberry Pi camera (HardwareWorld).
    """

    def read(self) -> np.ndarray:
        """Output: spike-compatible array for the target region."""
        ...

    def target_region(self) -> str:
        """Destination region in the brain, e.g. 'thalamus_visual' or 'reward_pathway'.

        Note: reward is also a “sense” and enters the brain through this path
        (dopaminergic neurons / RPE) — not directly from the environment.
        """
        ...


@runtime_checkable
class Actuator(Protocol):
    """Every output from the brain — motor spikes → action in the world."""

    def execute(self, action_vector: np.ndarray) -> None:
        """action_vector: spike pattern of motor areas.
        Each World decides how to interpret it
        (movement in a grid / GPIO servo motor / ...)."""
        ...


@runtime_checkable
class World(Protocol):
    """Every environment in which the brain lives."""

    def reset(self) -> None:
        """Reset to the initial episode state."""
        ...

    def step(self, dt_ms: int) -> None:
        """Advance the world physics by one tick.

        Note: no reward is returned — reward flows only through
        RewardSensor (reward contract ADR).
        """
        ...

    def sensors(self) -> list[Sensor]:
        """Currently active sensors of this world."""
        ...

    def actuators(self) -> list[Actuator]:
        """Actuators of this world that can be commanded."""
        ...

    def is_episode_done(self) -> bool:
        """Is the current episode finished?"""
        ...
