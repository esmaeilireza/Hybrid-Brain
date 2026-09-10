"""Month 5 acceptance harness v4 - conditioned avoidance (Week 18).

v3 PASSED with a scripted arousal gate. v4 replaces AROUSAL_GATE with
the real organ (regions/amygdala.py, audited):
  - place input: one-hot over the 20x20 grid at the agent position
  - dopamine gate: 0.0 at punishment events (conditioning opens),
    real dopamine level otherwise
  - amygdala.last_spikes.mean() -> EmotionEngine.amygdala_activation
    -> fear -> somatic marker. Fear now BUILDS across episodes as
    conditioning potentiates R's place pattern (three-factor rule).

Everything else from v3 unchanged: A/B ablation, isolated threat
channel (punishment never enters agent.learn), R from the greedy path,
competency gates, decision-seam eval.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from cognition.consolidation import Consolidation, ValueTable
from cognition.decision_making import DecisionMaker
from cognition.memory import EpisodicMemory
from cognition.reinforcement import RPEAgent
from core.neurotransmitters import NeuromodulatorSystem
from emotions.emotion_engine import EmotionEngine
from emotions.somatic import SomaticMarkerMap
from environment.grid_world_a import GridWorldA
from regions.amygdala import Amygdala

PARAMS = {
    "simulation": {"dt_ms": 1.0},
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
WORLD_CFG = {"environment": {"grid_size": 20, "max_steps": 300}}

START = (0, 0)
GRID = 20
PLACE_DIM = GRID * GRID
N_TRAIN = 80
N_EVAL = 10
COMPETENCY_MIN = 60
MOVES = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}


def one_hot(pos: tuple) -> np.ndarray:
    v = np.zeros(PLACE_DIM, dtype=float)
    v[pos[0] * GRID + pos[1]] = 1.0
    return v


def make_agent(somatic_on: bool) -> dict:
    vt = ValueTable(seed=1)
    agent = RPEAgent(vt, NeuromodulatorSystem(PARAMS))
    memory = EpisodicMemory()
    visits: dict = {}
    markers = SomaticMarkerMap() if somatic_on else None
    dm = DecisionMaker(vt, visits=visits, somatic=markers)
    return {"world": None, "agent": agent, "memory": memory,
            "engine": EmotionEngine(),
            "amgd": Amygdala(n_neurons=50, place_dim=PLACE_DIM,
                             params=PARAMS, seed=3),
            "markers": markers, "dm": dm,
            "visits": visits, "r_visits": 0,
            "cons": Consolidation(memory, vt)}


def neighbor_of(pos: tuple, action: int) -> tuple:
    d = MOVES[action]
    return (pos[0] + d[0], pos[1] + d[1])


def train_episode(sim: dict, rng, ep: int, punish_cell) -> bool:
    world, agent, engine, dm = (sim["world"], sim["agent"],
                                sim["engine"], sim["dm"])
    amgd = sim["amgd"]
    world.reset()
    world.agent = START
    steps, reached, goal = 0, False, world.goal
    eps = max(0.15, 0.6 * (1.0 - (ep - 1) / 40.0))
    while not world.is_episode_done() and steps < 300:
        pos = world.agent
        nb_pos = {a: neighbor_of(pos, a) for a in range(4)}
        nb_vals = {a: np.array(agent.value_signals(nb_pos[a]),
                               dtype=float) for a in range(4)}
        action, _ = dm.decide(pos, nb_vals, nb_pos)
        if rng.random() < eps:
            action = int(rng.integers(0, 4))

        world.pending_action = action
        world.step(1)
        steps += 1
        nxt = world.agent
        world_reward = float(world.sensors()[1].read()[0])

        novelty = 0.5 if nxt not in sim["visits"] else 0.0
        sim["visits"][nxt] = sim["visits"].get(nxt, 0) + 1
        dist_b = abs(pos[0] - goal[0]) + abs(pos[1] - goal[1])
        dist_a = abs(nxt[0] - goal[0]) + abs(nxt[1] - goal[1])

        punished = (punish_cell is not None and nxt == punish_cell)
        # threat channel only - agent.learn sees identical rewards
        # in both arms (the v3 isolation that made attribution causal)
        total = world_reward + novelty + 0.2 * (dist_b - dist_a)

        rpe = agent.learn(pos, action, total, next_position=nxt)
        agent.nt.step()

        # --- the real organ in the loop (v4) ---
        spikes = amgd.step(
            one_hot(nxt),
            dopamine_level=0.0 if punished else agent.nt.dopamine.level,
        )
        activation = float(spikes.mean())

        state = engine.step(
            rpe=rpe,
            amygdala_activation=activation,
            distance_reduced=dist_a < dist_b,
            pos_key=nxt,
            current_cell_value=-1.0 if punished else 0.0)
        if sim["markers"] is not None:
            if punished:
                sim["markers"].record(nxt, punishment=-1.0,
                                      arousal=state["arousal"])
            else:
                sim["markers"].note_safe_visit(nxt)
        sim["memory"].record(nxt, action, reward=rpe,
                             dopamine=agent.nt.dopamine.level,
                             place_pattern=one_hot(nxt).astype(np.uint8),
                             arousal=state["arousal"])
        if world.agent == goal:
            reached = True
    return reached


def greedy_eval(sim: dict, punish_cell) -> None:
    """Deterministic, NO learning, decided through the DecisionMaker
    seam so the marker's read-time bias is active."""
    world, dm, agent = sim["world"], sim["dm"], sim["agent"]
    world.reset()
    world.agent = START
    steps, goal = 0, world.goal
    while not world.is_episode_done() and steps < 300:
        pos = world.agent
        nb_pos = {a: neighbor_of(pos, a) for a in range(4)}
        nb_vals = {a: np.array(agent.value_signals(nb_pos[a]),
                               dtype=float) for a in range(4)}
        action, _ = dm.decide(pos, nb_vals, nb_pos)
        world.pending_action = action
        world.step(1)
        steps += 1
        nxt = world.agent
        if punish_cell is not None and nxt == punish_cell:
            sim["r_visits"] += 1
        if nxt == goal:
            break


def find_path_cell() -> tuple:
    """Probe: train punishment-free, greedy-eval, pick the most-visited
    non-terminal cell ON the deterministic path."""
    sim = make_agent(False)
    sim["world"] = GridWorldA(WORLD_CFG, seed=11)
    rng = np.random.default_rng(7)
    for ep in range(1, 81):
        train_episode(sim, rng, ep, punish_cell=None)
    path_counts: dict = {}
    for _ in range(N_EVAL):
        greedy_eval(sim, None)
        # track path via visits delta (greedy_eval counts r only):
    # re-count properly by tracking in a dedicated dict:
    sim2 = make_agent(False)
    sim2["world"] = GridWorldA(WORLD_CFG, seed=11)
    # reuse the trained value table through a fresh agent sharing it:
    sim2["agent"] = sim["agent"]
    sim2["dm"] = DecisionMaker(sim["agent"].values, visits={},
                               somatic=None)
    path_counts = {}
    world = sim2["world"]
    for _ in range(N_EVAL):
        world.reset()
        world.agent = START
        steps, goal = 0, world.goal
        while not world.is_episode_done() and steps < 300:
            pos = world.agent
            nb_pos = {a: neighbor_of(pos, a) for a in range(4)}
            nb_vals = {a: np.array(sim["agent"].value_signals(nb_pos[a]),
                                   dtype=float) for a in range(4)}
            action, _ = sim2["dm"].decide(pos, nb_vals, nb_pos)
            world.pending_action = action
            world.step(1)
            steps += 1
            nxt = world.agent
            path_counts[nxt] = path_counts.get(nxt, 0) + 1
            if nxt == goal:
                break
    candidates = {p: c for p, c in path_counts.items()
                  if p not in (START, sim["world"].goal)}
    return max(candidates, key=candidates.get)


def main() -> None:
    r_cell = find_path_cell()
    print(f"region R (greedy-path corridor cell): {r_cell}")

    results = {}
    for label, somatic_on in (("OFF", False), ("ON", True)):
        sim = make_agent(somatic_on)
        sim["world"] = GridWorldA(WORLD_CFG, seed=11)
        rng = np.random.default_rng(42)
        hits = sum(train_episode(sim, rng, e, r_cell)
                   for e in range(1, N_TRAIN + 1))
        sim["cons"].sleep(replay_k=50)
        eval_goals = 0
        for _ in range(N_EVAL):
            greedy_eval(sim, r_cell)
            if sim["world"].agent == sim["world"].goal:
                eval_goals += 1
        results[label] = {"train": hits, "eval": eval_goals,
                          "r": sim["r_visits"]}
        print(f"somatic {label}: train goals {hits}/{N_TRAIN}, "
              f"eval goals {eval_goals}/{N_EVAL}, "
              f"R-visits(eval): {sim['r_visits']}")

    if any(r["train"] < COMPETENCY_MIN for r in results.values()):
        print(f"\nINVALID: navigation competency not reached "
              f"(<{COMPETENCY_MIN}/{N_TRAIN})")
        sys.exit(1)
    if any(r["eval"] < 8 for r in results.values()):
        print("\nINVALID: eval navigation degraded - differential "
              "not interpretable")
        sys.exit(1)

    off, on = results["OFF"]["r"], results["ON"]["r"]
    print(f"\navoidance differential: ON={on} vs OFF={off} visits to R")
    if off == 0:
        print("INCONCLUSIVE: baseline never visits R")
    elif on <= 0.5 * off:
        print("ACCEPTED: >50% avoidance attributable to somatic markers "
              "(amygdala-driven fear, real organ)")
    else:
        print("FAILED: marker effect below the 50% criterion")
        sys.exit(1)


if __name__ == "__main__":
    main()
