"""2D activity heatmap - the first debugging eye of the project.
Plot region activity over time; render only on demand (never block
the simulation loop - that rule arrives with the 3D layer)."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # headless-safe; save to file, never plt.show() mid-sim
import matplotlib.pyplot as plt
import numpy as np


def plot_region_activity(
    activity_log: dict[str, np.ndarray],
    dt_ms: float,
    out_path: str = "data/logs/activity.png",
    title: str = "Region activity (spikes/tick, smoothed)",
) -> str:
    """activity_log: {region_name: 1D array of per-tick spike counts}."""
    fig, ax = plt.subplots(figsize=(12, 4 + 0.4 * len(activity_log)))
    t_s = np.arange(next(iter(activity_log.values())).size) * dt_ms / 1000.0
    for name, counts in activity_log.items():
        # convolve(mode='same') returns max(len(x), len(kernel)) -
        # short traces (fast episodes) must use a smaller kernel
        k = min(50, len(counts))
        kernel = np.ones(k) / k
        smoothed = np.convolve(counts, kernel, mode="same")
        ax.plot(t_s, smoothed, label=name, linewidth=1.2)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("spikes/tick (smoothed)")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    return out_path
