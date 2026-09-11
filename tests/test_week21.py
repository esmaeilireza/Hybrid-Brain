# Week 21 - vision pipeline + VisualWorld contract.
# Acceptance (ADR-017): same object nearby poses corr > 0.9;
# different objects corr < 0.5 when pretrained (looser otherwise).
import time

import numpy as np
import pytest

from environment.visual_world import VisualWorld
from perception.vision import VisionSystem


@pytest.fixture(scope="module")
def vision():
    return VisionSystem(seed=3)


@pytest.fixture(scope="module")
def world():
    w = VisualWorld(seed=1)
    w.reset()
    return w


def test_world_protocol_compliance(world):
    from core.interfaces import World
    assert isinstance(world, World)
    regions = {s.target_region() for s in world.sensors()}
    assert regions == {"visual_cortex", "reward_pathway"}
    assert len(world.actuators()) == 1


def test_render_shape_and_speed(world):
    t0 = time.perf_counter()
    for _ in range(5):
        frame = world.render()
    dt = time.perf_counter() - t0
    assert frame.shape[2] == 3 and frame.shape[0] > 200
    assert dt / 5 < 1.0


def test_same_object_nearby_poses_correlate(vision, world):
    world.reset()
    world.pos = np.array([2.0, 2.5])
    world.yaw = 0.3
    f1 = vision.extract(world.render())
    world.pos = np.array([2.2, 2.3])
    world.yaw = 0.45
    f2 = vision.extract(world.render())
    if f1 is None:
        pytest.skip("torch unavailable")
    assert f1.shape == (576,)
    corr = float(np.corrcoef(f1, f2)[0, 1])
    assert corr > 0.9, f"same-object stability: corr={corr:.3f}"


def test_different_objects_discriminate(vision, world):
    world.reset()
    world.pos = np.array([2.0, 2.5])
    world.yaw = 0.3
    f_cube = vision.extract(world.render())
    world.pos = np.array([4.0, 2.0])
    world.yaw = 0.0
    f_sphere = vision.extract(world.render())
    if f_cube is None:
        pytest.skip("torch unavailable")
    corr = float(np.corrcoef(f_cube, f_sphere)[0, 1])
    assert corr < 0.9, "objects indistinguishable even loosely"
    if vision.weights_source == "pretrained":
        assert corr < 0.5, f"weak discrimination: {corr:.3f}"


def test_projection_drive_in_fi_range(vision):
    f = np.zeros(576)
    f[:16] = 1.0
    drive = vision.project(f / np.linalg.norm(f))
    assert drive.shape == (100,)
    active = drive[drive > 0]
    # dilution checklist: active drives >= 0.93 (docstring minimum)
    if active.size:
        assert active.mean() >= 0.93
