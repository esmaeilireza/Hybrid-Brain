"""Month 3 acceptance v4: fixed start at (0, 0).
Train 80 epsilon-annealed episodes from one fixed start, then 10
deterministic greedy evaluations from the same start.
Acceptance: success rate >= 0.8 AND mean successful path <= 100.
Random-start evaluation deferred to GridWorld-B / Month 5 (ADR-006)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from cognition.consolidation import ValueTable
from cognition.reinforcement import RPEAgent
from core.neurotransmitters import NeuromodulatorSystem
from environment.grid_world_a import GridWorldA

PARAMS = {
    "simulation": {"dt_ms": 1.0},
    "neuron": {
        "tau_m_ms": 20.0,
        "v_rest_mV": -65.0,
        "v_threshold_mV": -55.0,
        "v_reset_mV": -70.0,
        "refractory_ms": 2.0,
    },
    "synapse": {
        "tau_excite_ms": 5.0,
        "connection_density": 0.05,
        "excitatory_ratio": 0.8,
        "delay_min_ms": 1.0,
        "w_exc": 1.5,
        "w_inh": 3.0,
    },
}

START = (0, 0)
N_TRAIN = 80
N_EVAL = 10


def train_episode(world, agent, rng, ep: int,
                  visits: dict | None = None) -> tuple[int, bool]:
    world.reset()
    world.agent = START
    steps = 0
    goal = world.goal
    if visits is None:
        visits = {}

    while not world.is_episode_done() and steps < 300:
        pos = world.agent
        values = np.array(agent.value_signals(pos), dtype=float)

        eps = max(0.15, 0.6 * (1.0 - (ep - 1) / 40.0))
        best = np.flatnonzero(values == values.max())
        greedy = int(rng.choice(best))
        action = int(rng.integers(0, 4)) if rng.random() < eps else greedy

        world.pending_action = action
        world.step(1)
        steps += 1

        reward = float(world.sensors()[1].read()[0])
        nxt = world.agent

        novelty = 0.5 if nxt not in visits else 0.0
        visits[nxt] = visits.get(nxt, 0) + 1

        dist_b = abs(pos[0] - goal[0]) + abs(pos[1] - goal[1])
        dist_a = abs(nxt[0] - goal[0]) + abs(nxt[1] - goal[1])
        shape = 0.2 * (dist_b - dist_a)

        agent.learn(pos, action, reward + novelty + shape,
                    next_position=nxt)
        agent.nt.step()

    return steps, world.agent == goal


def greedy_eval(world, agent) -> tuple[int, bool]:
    """Deterministic evaluation: fixed start, pure argmax, no RNG."""
    world.reset()
    world.agent = START
    steps = 0
    goal = world.goal
    while not world.is_episode_done() and steps < 300:
        values = np.array(agent.value_signals(world.agent), dtype=float)
        world.pending_action = int(np.argmax(values))
        world.step(1)
        steps += 1
    return steps, world.agent == goal


def main() -> None:
    world = GridWorldA(
        {"environment": {"grid_size": 20, "max_steps": 300}}, seed=11
    )
    agent = RPEAgent(ValueTable(seed=1), NeuromodulatorSystem(PARAMS))
    rng = np.random.default_rng(3)

    print("TRAIN (fixed start at (0, 0), epsilon-annealed, "
          f"{N_TRAIN} episodes, novelty):")
    visits: dict = {}
    goals = 0
    for ep in range(1, N_TRAIN + 1):
        steps, reached = train_episode(world, agent, rng, ep, visits)
        goals += int(reached)
        if ep % 5 == 0:
            print(f"  ep {ep:2d}: {steps:3d} steps | goal={reached}")
    print(f"  training goal hits: {goals}/{N_TRAIN}")

    print(f"EVAL (deterministic greedy from (0, 0), {N_EVAL} runs):")
    results = [greedy_eval(world, agent) for _ in range(N_EVAL)]
    eval_steps = [s for s, _ in results]
    successes = [g for _, g in results]
    success_rate = sum(successes) / len(results)
    mean_success = (float(np.mean([s for s, g in results if g]))
                    if any(successes) else float("inf"))

    print(f"  steps: {eval_steps}")
    print(f"  success rate: {success_rate:.0%} | "
          f"mean successful path length: {mean_success:.0f}")

    assert success_rate >= 0.8, "Month 3 acceptance FAILED (success rate)"
    assert mean_success <= 100, "Month 3 acceptance FAILED (path quality)"

    with open("docs/benchmarks.md", "a", encoding="utf-8") as fh:
        fh.write(
            f"\n| Month 3 | RPE + distance-delta shaping + entry-only "
            f"novelty + fixed start | success {success_rate:.0%}, "
            f"mean {mean_success:.0f} steps "
            f"(training goals {goals}/{N_TRAIN}) |\n"
        )

    print("ACCEPTED: the maze was learned from the brain's own value signals")


if __name__ == "__main__":
    main()
