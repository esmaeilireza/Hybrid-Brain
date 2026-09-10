"""Week 7 - the region graph routes and flows."""
import numpy as np

from core.connectome import Brain
from environment.grid_world_a import GridWorldA

PARAMS = {
    "simulation": {"dt_ms": 1.0, "seed": 42},
    "neuron": {
        "tau_m_ms": 20.0, "v_rest_mV": -65.0, "v_threshold_mV": -55.0,
        "v_reset_mV": -70.0, "refractory_ms": 2.0,
    },
    "synapse": {
        "tau_excite_ms": 5.0, "connection_density": 0.05,
        "excitatory_ratio": 0.8, "delay_min_ms": 1.0,
        "w_exc": 1.5, "w_inh": 3.0,
    },
}


def test_brain_cycles_produce_flow() -> None:
    world = GridWorldA({"environment": {"grid_size": 20, "max_steps": 300}}, seed=9)
    world.reset()
    brain = Brain(world, PARAMS)
    th, cx = [], []
    for _ in range(30):
        action, counts = brain.step()
        world.pending_action = action
        world.step(1)
        th.append(counts["thalamus"])
        cx.append(counts["cortex"])
    assert sum(th) > 50, "thalamus silent"
    assert sum(cx) > 50, "cortex silent"
