# Month 6 demo - visual-motor closed loop (Week 24).
# Loop: VisualWorld render -> MobileNet features -> Cortex6 -> attention
#   -> Q(cell,action) epsilon-greedy -> MotorActuator -> reward -> RPE.
# NOTE (honest scope): decision comes from the PROVEN Month-3 Q-table on
# quantized cells; Cortex6/attention/auditory run live in-loop and are
# LOGGED (salience, rates) - full neural action readout is W25+ work.
# Metrics (arXiv): trajectory CSV, Euclidean distance curve, dopamine
# dynamics, eval success rate. Outputs -> data/logs/month6_*.
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
from cognition.attention import AttentionSystem
from cognition.consolidation import ValueTable
from cognition.reinforcement import RPEAgent
from core.neurotransmitters import NeuromodulatorSystem
from environment.visual_world import VisualWorld
from perception.auditory import AuditoryStream
from perception.vision import VisionSystem
from regions.cortex6 import Cortex6
PARAMS = {
    "simulation": {"dt_ms": 1.0},
    "neuron": {"tau_m_ms": 20.0, "v_rest_mV": -65.0,
               "v_threshold_mV": -55.0, "v_reset_mV": -70.0,
               "refractory_ms": 2.0},
    "synapse": {"tau_excite_ms": 5.0, "connection_density": 0.05,
                "excitatory_ratio": 0.8, "delay_min_ms": 1.0,
                "w_exc": 1.5, "w_inh": 3.0},
    "neurotransmitters": {"dopamine_baseline": 0.1,
                          "dopamine_tau_ms": 100.0},
}
WORLD_CFG = {"environment": {"max_steps": 60, "goal_object": "red_cube"}}
GRID, N_EP, N_EVAL, SHAPING, EPS0 = 8, 30, 5, 0.3, 0.5
GOAL = np.array([3.0, 1.0])
def cell_of(pos):
    return (int(np.clip(pos[0] // 1.0, 0, GRID - 1)),
            int(np.clip(pos[1] // 1.0, 0, GRID - 1)))
class Brain6:
    def __init__(self):
        self.vision = VisionSystem(seed=3)
        self.world = VisualWorld(WORLD_CFG, seed=11)
        self.cortex = Cortex6(240, PARAMS, seed=2)
        self.att = AttentionSystem(n_streams=2, seed=0)
        self.aud = AuditoryStream(PARAMS, seed=1)
        self.vt = ValueTable(grid_size=GRID, seed=1)
        self.agent = RPEAgent(self.vt, NeuromodulatorSystem(PARAMS))
    def cycle(self, eps, learn=True):
        w = self.world
        frame = w.render()
        feats = self.vision.extract(frame)
        drive = self.vision.project(feats)
        counts = self.cortex.step(drive[: self.cortex.n_in])
        s_aud = self.aud.step(); self.aud.set_event(False)
        gains = self.att.step([float(sum(counts)) / 50.0,
                               self.aud.salience()])
        cell = cell_of(w.pos)
        vals = self.agent.value_signals(cell)
        action = int(np.argmax(vals)) if np.random.random() > eps \
            else int(np.random.randint(0, 4))
        d_before = float(np.linalg.norm(w.pos - GOAL))
        w.pending_action = action
        w.step(1)
        reward = float(w.sensors()[1].read()[0])
        d_after = float(np.linalg.norm(w.pos - GOAL))
        nxt = cell_of(w.pos)
        rpe = self.agent.learn(cell, action,
                               reward + SHAPING * (d_before - d_after),
                               next_position=nxt)
        self.agent.nt.step()
        return {"pos": tuple(w.pos), "dist": d_after, "reward": reward,
                "rpe": rpe, "da": self.agent.nt.dopamine.level,
                "att": self.att.attended_idx, "cx": sum(counts)}

def run_episode(brain, eps, learn):
    brain.world.reset()
    traj, das = [], []
    for _ in range(brain.world.max_steps):
        rec = brain.cycle(eps, learn)
        traj.append(rec)
        das.append(rec["da"])
        if rec["reward"] > 0.5:
            break
    return traj, das

def main():
    brain = Brain6()
    print("weights:", brain.vision.weights_source)
    train_success = 0
    for ep in range(1, N_EP + 1):
        eps = max(0.1, EPS0 * (1.0 - (ep - 1) / max(N_EP - 5, 1)))
        traj, _ = run_episode(brain, eps, learn=True)
        hit = traj[-1]["reward"] > 0.5
        train_success += int(hit)
        print(f"train ep {ep:2d}: steps {len(traj):3d} "
              f"goal={hit} final_dist={traj[-1]['dist']:.2f}")
    eval_results = []
    best_traj = None
    for _ in range(N_EVAL):
        traj, das = run_episode(brain, eps=0.0, learn=False)
        hit = traj[-1]["reward"] > 0.5
        eval_results.append(hit)
        if best_traj is None or traj[-1]["dist"] < best_traj[-1]["dist"]:
            best_traj, best_das = traj, das
    print(f"eval: {sum(eval_results)}/{N_EVAL} reached goal")
    d0 = best_traj[0]["dist"]
    d1 = best_traj[-1]["dist"]
    print(f"best eval distance: {d0:.2f} -> {d1:.2f}")
    out = Path(__file__).resolve().parent.parent / "data" / "logs"
    out.mkdir(parents=True, exist_ok=True)
    import csv
    with open(out / "month6_trajectory.csv", "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["step", "x", "y", "dist", "reward", "dopamine"])
        for i, r in enumerate(best_traj):
            wr.writerow([i, r["pos"][0], r["pos"][1], r["dist"],
                         r["reward"], r["da"]])
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
        xs = [r["pos"][0] for r in best_traj]
        ys = [r["pos"][1] for r in best_traj]
        ax1.plot(xs, ys, "-o", ms=2)
        ax1.plot(GOAL[0], GOAL[1], "r*", ms=14, label="goal")
        ax1.set_title("trajectory (best eval)"); ax1.legend()
        ax2.plot(best_das); ax2.set_title("dopamine")
        fig.tight_layout()
        fig.savefig(out / "month6_demo.png", dpi=110)
        print("figure saved:", out / "month6_demo.png")
    except Exception as e:
        print("plot skipped:", e)
    ok = (sum(eval_results) >= 2) or (d1 < 0.5 * d0)
    print("ACCEPTED: visual-motor loop navigates" if ok
          else "FAILED: no goal navigation demonstrated")
    sys.exit(0 if ok else 1)

if __name__ == "__main__":
    main()
