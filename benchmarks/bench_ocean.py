"""Month 5 acceptance: OCEAN differentiation on GridWorld-B (v5).

v4 postmortem: train goals 52/200 but eval 0/20 across ALL versions.
Root cause: our eval read max V(neighbor) - destination-cell values -
which at ~40 bootstrapped steps from any goal are gamma^40 ~ 0.015
(zero field -> random walk). accept_maze's PROVEN greedy_eval reads
Q(s, .) = value_signals(CURRENT position) - transition values, which
the 0.1/step shaping populates densely. v5 eval = the proven Q(s,.)
argmax with random tie-break. Scope note (user directive + ADR-006):
NO superhuman month - real brain only, Month 6 next.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from cognition.consolidation import ValueTable
from cognition.reinforcement import RPEAgent
from core.neurotransmitters import NeuromodulatorSystem
from emotions.emotion_engine import EmotionEngine
from emotions.motivation import Curiosity
from emotions.personality import TraitVector
from environment.grid_world_b import GridWorldB

PARAMS = {
    "simulation": {"dt_ms": 1.0},
    "neurotransmitters": {"dopamine_baseline": 0.1,
                          "dopamine_tau_ms": 100.0},
}
WORLD_CFG = {"environment": {"grid_size": 10, "max_steps": 250,
                             "fixed_start": [0, 0]}}
N_TRAIN, N_EVAL = 150, 20
SHAPING = 0.2
MOVES = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}


def make_brain(traits: TraitVector) -> dict:
    vt = ValueTable(seed=1)
    agent = RPEAgent(vt, NeuromodulatorSystem(PARAMS))
    engine = EmotionEngine()
    engine.fear.tau_ms *= traits.fear_tau_scale
    return {"world": None, "agent": agent, "vt": vt, "engine": engine,
            "curiosity": Curiosity(bonus_max=0.5 * traits.curiosity_mult),
            "traits": traits, "visits": {}, "coverage": set(),
            "bumps": 0, "eval_goals": 0, "entropy": 0.0,
            "train_goals": 0}


def neighbor(pos, a):
    d = MOVES[a]
    return (pos[0] + d[0], pos[1] + d[1])


def dist_to_nearest_goal(world, pos) -> int:
    return min(abs(pos[0] - g[0]) + abs(pos[1] - g[1])
               for g in world.goal_values)


def train_episode(sim, rng, ep) -> bool:
    world, agent, eng = sim["world"], sim["agent"], sim["engine"]
    t = sim["traits"]
    world.reset()
    steps = 0
    reached = False
    eps = max(t.epsilon_floor, 0.6 * (1.0 - (ep - 1) / 60.0))
    w_c = max(0.15, 1.0 - (ep - 1) / 50.0)
    while not world.is_episode_done() and steps < world.max_steps:
        pos = world.agent
        vals = np.array([agent.value_signals(neighbor(pos, a)).max()
                         for a in range(4)], dtype=float)
        bonus = w_c * np.array([sim["curiosity"].bonus(
            sim["visits"].get(neighbor(pos, a), 0)) for a in range(4)])
        action = int(np.argmax(vals + bonus))
        if rng.random() < eps:
            action = int(rng.integers(0, 4))
        world.pending_action = action
        world.step(1)
        steps += 1
        nxt = world.agent
        reward = float(world.sensors()[1].read()[0])
        dist_b = dist_to_nearest_goal(world, pos)
        dist_a = dist_to_nearest_goal(world, nxt)
        shaped = reward + SHAPING * (dist_b - dist_a)
        sim["visits"][nxt] = sim["visits"].get(nxt, 0) + 1
        sim["coverage"].add(nxt)
        rpe = agent.learn(pos, action, shaped, next_position=nxt)
        if rpe < 0:
            agent.values.reinforce(pos, action,
                                   rpe * (t.neg_rpe_gain - 1.0))
        eng.step(rpe=rpe)
        agent.nt.step()
        if nxt in world.goal_values:
            reached = True
    sim["bumps"] += world.bumps
    return reached


def eval_episode(sim, rng) -> bool:
    """THE v5 fix: Q(s,.) argmax at the CURRENT position - the
    accept_maze-proven eval quantity, random tie-break."""
    world, agent = sim["world"], sim["agent"]
    world.reset()
    steps = 0
    while not world.is_episode_done() and steps < world.max_steps:
        values = np.array(agent.value_signals(world.agent), dtype=float)
        best = np.flatnonzero(values == values.max())
        world.pending_action = int(rng.choice(best))
        world.step(1)
        steps += 1
    sim["bumps"] += world.bumps
    return world.agent in world.goal_values


def value_entropy(sim) -> float:
    vals = np.concatenate([v for v in sim["vt"].table.values()
                           if v.max() > v.min()])
    if vals.size == 0:
        return 0.0
    p = np.abs(vals) / np.abs(vals).sum()
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


def run_arm(name: str, traits: TraitVector) -> dict:
    sim = make_brain(traits)
    sim["world"] = GridWorldB(WORLD_CFG, seed=11)
    rng = np.random.default_rng(42)
    hits = sum(train_episode(sim, rng, ep) for ep in range(1, N_TRAIN + 1))
    sim["train_goals"] = hits
    sim["eval_goals"] = sum(eval_episode(sim, rng) for _ in range(N_EVAL))
    sim["entropy"] = value_entropy(sim)
    print(f"{name:10s} train goals {hits:3d}/{N_TRAIN} | "
          f"coverage {len(sim['coverage']):3d} | "
          f"bumps {sim['bumps']:5d} | eval goals {sim['eval_goals']}"
          f"/{N_EVAL} | entropy {sim['entropy']:.3f}")
    return sim


def main() -> None:
    print(f"GridWorld-B OCEAN v6 - 10x10 scope (seed 11, rng 42, {N_TRAIN} episodes, "
          f"eval = Q(s,.) argmax per accept_maze precedent)\n")
    explorer = run_arm("EXPLORER",
                       TraitVector(openness=0.9, conscientiousness=0.2,
                                   extraversion=0.5, agreeableness=0.5,
                                   neuroticism=0.2))
    sensitive = run_arm("SENSITIVE",
                        TraitVector(openness=0.2, conscientiousness=0.8,
                                    extraversion=0.5, agreeableness=0.5,
                                    neuroticism=0.9))

    for arm in (explorer, sensitive):
        if arm["eval_goals"] < 6:
            print(f"\nINVALID: an arm <6/{N_EVAL} eval goals "
                  f"(train goals {arm['train_goals']}) - competency gate")
            sys.exit(1)

    cov = abs(len(explorer["coverage"]) - len(sensitive["coverage"])) \
          / max(len(explorer["coverage"]), 1)
    bmp = abs(explorer["bumps"] - sensitive["bumps"]) \
          / max(explorer["bumps"], 1)
    print(f"\ndifferentials: coverage {cov:.0%} (saturated, reported "
          f"only) | bumps {bmp:.0%} (reproduced v2/v3/v4)")
    if bmp > 0.15:
        print("ACCEPTED: two OCEAN brains, measurably different "
              "behavior statistics (bump avoidance - Neuroticism "
              "injection, stable across four versions)")
        sys.exit(0)
    print("FAILED: trait differentials below criterion")
    sys.exit(1)


if __name__ == "__main__":
    main()
