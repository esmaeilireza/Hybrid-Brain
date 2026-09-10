"""Week 6 demo: the agent moves by ITS OWN basal ganglia decision."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from regions.basal_ganglia import BasalGanglia
from regions.hippocampus import PlaceCellPopulation

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


def main() -> None:
    from environment.grid_world_a import _DXY, GridWorldA

    world = GridWorldA({"environment": {"grid_size": 20, "max_steps": 500}}, seed=5)
    world.reset()
    place = PlaceCellPopulation(200, grid_size=20, params=PARAMS, seed=2)
    bg = BasalGanglia(4, PARAMS, seed=1)

    goal = np.array(world.goal)
    steps = 0
    while not world.is_episode_done() and steps < 500:
        r, c = world.agent
        values = np.zeros(4)
        for a, (dr, dc) in _DXY.items():
            nr = min(max(r + dr, 0), 19)
            nc = min(max(c + dc, 0), 19)
            values[a] = 1.0 / (1.0 + abs(goal[0] - nr) + abs(goal[1] - nc))
        world.pending_action = bg.decide(values)
        world.step(1)
        place.step(world.agent)
        steps += 1

    print(f"Episode finished in {steps} steps; at goal: {world.agent == world.goal}")
    print("(random walk on 20x20 averages ~200+ steps to a corner)")


if __name__ == "__main__":
    main()
