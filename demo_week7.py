"""Week 7 demo: full signal flow, one heatmap.
GridWorld -> Thalamus -> Cortex -> (Hippocampus) -> BG -> action.
Acceptance: the heatmap shows the causal chain - thalamic activity
precedes cortical response in every cycle."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from core.connectome import Brain
from environment.grid_world_a import GridWorldA
from visualization.heatmap_2d import plot_region_activity

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
    world = GridWorldA({"environment": {"grid_size": 20, "max_steps": 300}}, seed=5)
    world.reset()
    brain = Brain(world, PARAMS)

    log = {"thalamus": [], "cortex": [], "hippocampus": []}
    steps = 0
    while not world.is_episode_done() and steps < 300:
        action, counts = brain.step()
        world.pending_action = action
        world.step(1)
        # hippocampus runs every cycle (place coding is continuous)
        s_hc = brain.hippocampus.step(world.agent)
        log["thalamus"].append(counts["thalamus"])
        log["cortex"].append(counts["cortex"])
        log["hippocampus"].append(int(s_hc.sum()))
        steps += 1

    out = plot_region_activity(
        {k: np.array(v, float) for k, v in log.items()},
        dt_ms=50.0, out_path="data/logs/signal_flow.png",
        title="Month 2 acceptance: env -> thalamus -> cortex -> PFC/BG flow",
    )
    print(f"Episode: {steps} steps, at goal: {world.agent == world.goal}")
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
