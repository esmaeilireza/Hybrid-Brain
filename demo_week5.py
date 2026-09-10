"""Week 5 demo: GridWorld -> Thalamus -> Hippocampus -> Heatmap.

Watch an agent wander the grid while two brain regions encode what it
perceives and where it is. Outputs data/logs/activity.png and
data/logs/place_raster.png - the first picture of this brain thinking."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from environment.grid_world_a import ACTIONS
from regions.hippocampus import PlaceCellPopulation
from regions.thalamus import Thalamus
from visualization.heatmap_2d import plot_region_activity

PARAMS = {
    "simulation": {"dt_ms": 1.0, "seed": 42},
    "neuron": {
        "tau_m_ms": 20.0, "v_rest_mV": -65.0, "v_threshold_mV": -55.0,
        "v_reset_mV": -70.0, "refractory_ms": 2.0,
    },
}


def main() -> None:
    world = GridWorldA({"environment": {"grid_size": 20, "max_steps": 100_000}}, seed=3)
    world.reset()
    thalamus = Thalamus(150, sensor_dim=400, params=PARAMS, seed=1)
    place = PlaceCellPopulation(200, grid_size=20, params=PARAMS, seed=2)

    rng = np.random.default_rng(7)
    n_ticks, log_every = 20_000, 1
    th_log, hc_log, positions = [], [], []

    for tick in range(n_ticks):
        if tick % 50 == 0:                      # a "thought" every 50 ms
            obs = world.sensors()[0].read()
            s_th = thalamus.step(obs)
            pc = place.step(world.agent)
            world.pending_action = int(rng.integers(0, 4))  # random walk
            world.step(1)
            positions.append(world.agent)
            th_log.append(int(s_th.sum()))
            hc_log.append(int(pc.sum()))

    out1 = plot_region_activity(
        {"thalamus": np.array(th_log, float), "hippocampus": np.array(hc_log, float)},
        dt_ms=50.0, out_path="data/logs/activity.png",
        title="Week 5: agent wandering - region activity",
    )
    raster = np.zeros((200, len(positions)))
    for i, (r, c) in enumerate(positions):
        for j, (cr, cc) in enumerate(place.centers):
            if (cr, cc) == (r, c):
                raster[j, i] = 1
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.imshow(raster, aspect="auto", interpolation="nearest")
    ax.set_xlabel("sample"); ax.set_ylabel("place cell #")
    ax.set_title("Place-cell raster (active cell follows agent)")
    fig.tight_layout()
    fig.savefig("data/logs/place_raster.png", dpi=110)
    plt.close(fig)

    print(f"Saved: {out1}")
    print("Saved: data/logs/place_raster.png")
    print(f"Samples: {len(positions)}, positions visited: {len(set(positions))}/400")


if __name__ == "__main__":
    from environment.grid_world_a import GridWorldA
    main()
