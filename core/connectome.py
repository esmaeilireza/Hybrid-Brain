"""Brain - the region graph. Owns the World and all regions; routes
sensor output by target_region() per the contract. The world never
knows about regions; regions never know about the world."""
from __future__ import annotations

import numpy as np
from scipy import sparse

from regions.basal_ganglia import BasalGanglia
from regions.cortex import Cortex
from regions.hippocampus import PlaceCellPopulation
from regions.thalamus import Thalamus


class Brain:
    def __init__(self, world, params: dict) -> None:
        self.world = world
        grid = 20
        self.thalamus = Thalamus(100, sensor_dim=grid * grid, params=params, seed=1)
        self.cortex = Cortex(200, params, seed=2)
        self.hippocampus = PlaceCellPopulation(200, grid, params, seed=3)
        self.bg = BasalGanglia(4, params, seed=4)

        # Thalamus(100) -> Cortex(200 E): sparse convergent projection,
        # gain calibrated so a thalamic neuron's spikes drive cortical
        # partners above threshold (f-I curve, Week-2 calibration rule).
        rng = np.random.default_rng(10)
        n_th, n_cx = 100, 200
        k = 8   # each cortical neuron reads 8 thalamic neurons
        rows = np.repeat(np.arange(n_cx), k)
        cols = rng.integers(0, n_th, size=n_cx * k)
        self.th2cx = sparse.csr_matrix(
            (np.full(n_cx * k, 8.0), (rows, cols)),
            shape=(n_cx, n_th),
        )
        # route map built from contract metadata, not hardcoded names
        self._routes = {
            s.target_region(): s for s in world.sensors()
        }

    def step(self, dt_ms: int = 1) -> tuple[int, dict[str, int]]:
        """One cognitive cycle (~every 50 ms in demos). Returns
        (chosen_action, per-region spike counts)."""
        counts: dict[str, int] = {}

        # sensory relay
        grid_sensor = self._routes.get("thalamus_spatial")
        obs = grid_sensor.read() if grid_sensor else np.zeros(400)
        # A cognitive cycle = ~50 ms of neural time. The thalamus runs a
        # settle window per cycle (BG decide() taught us: 1 tick after a
        # sensory change sees only resting neurons). Spikes accumulate
        # across the window; downstream regions consume the total.
        s_th_acc = np.zeros(100, dtype=bool)
        s_th_rate = np.zeros(100)
        th_count = 0
        for _ in range(15):
            s = self.thalamus.step(obs)
            s_th_acc |= s
            s_th_rate += s.astype(float)
            th_count += int(s.sum())
        s_th = s_th_acc
        s_th_signal = s_th_rate   # rate-coded: counts over the window
        counts["thalamus"] = th_count

        # cortical processing of thalamic pattern (drive = thalamic rates)
        # Timescale law (same fix as thalamus/BG): one tick of Euler
        # moves v only 5% toward steady state - a single-tick pulse
        # cannot cross a 10 mV gap regardless of amplitude. The cortex
        # runs the same 15-tick window with the afferent drive HELD.
        cx_input = self.th2cx @ s_th_signal
        s_cx_acc = np.zeros(self.cortex.n_exc, dtype=bool)
        cx_count = 0
        for _ in range(15):
            s = self.cortex.step(cx_input)
            s_cx_acc |= s
            cx_count += int(s.sum())
        s_cx = s_cx_acc
        counts["cortex"] = int(s_cx.sum())

        # spatial encoding
        s_hc = self.hippocampus.step(self.world.agent)
        counts["hippocampus"] = int(s_hc.sum())

        # decision: BG consumes cortical+hippocampal activity via a
        # placeholder value function (Month 3 replaces with learned values)
        r, c = self.world.agent
        goal = np.array(self.world.goal)
        values = np.zeros(4)
        for a, (dr, dc) in {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}.items():
            nr, nc = min(max(r + dr, 0), 19), min(max(c + dc, 0), 19)
            values[a] = 1.0 / (1.0 + abs(goal[0] - nr) + abs(goal[1] - nc))
        vmin, vmax = values.min(), values.max()
        if vmax > vmin:
            values = 0.2 + 0.7 * (values - vmin) / (vmax - vmin)
        action = self.bg.decide(values)
        counts["basal_ganglia"] = -1   # internal spikes not exposed; marker

        return action, counts
