"""Month 3 acceptance: maze solved by LEARNED values.
20 episodes. BG selects from ValueTable (hand-written distance function
deleted from the decision path). RPE learning per step, consolidation
sleep between episodes. Acceptance: ep20 steps < 50% of ep1 steps."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from cognition.consolidation import Consolidation, ValueTable
from cognition.memory import EpisodicMemory
from cognition.reinforcement import RPEAgent
from core.neurotransmitters import NeuromodulatorSystem
from environment.grid_world_a import GridWorldA

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


def run_episode(world, agent: RPEAgent, memory: EpisodicMemory,
                rng, ep: int, max_steps: int = 300) -> int:
    world.reset()
    steps = 0
    while not world.is_episode_done() and steps < max_steps:
        pos = world.agent
        values = agent.value_signals(pos)
        # epsilon-greedy: annealed exploration (softmax on flat values
        # was a disguised random walk - Week-6 relative-coding lesson)
        eps = max(0.05, 0.4 * (1.0 - (ep - 1) / 10.0))
        if rng.random() < eps:
            action = int(rng.integers(0, 4))
        else:
            action = int(np.argmax(values))

        world.pending_action = action
        world.step(1)
        steps += 1

        reward = world.sensors()[1].read()[0]     # reward via Sensor
        nxt = None if world.is_episode_done() else world.agent

        # Potential-based shaping (Ng et al. 1999): F = gamma*Phi(s')
        # - Phi(s), Phi = -manhattan distance to goal. Policy-invariant:
        # the value function is still LEARNED by the brain; shaping only
        # densifies the reward so the goal need not be found by chance
        # within 300 steps (20x20 random-walk hitting time >> 300).
        goal = world.goal
        phi_before = -(abs(pos[0] - goal[0]) + abs(pos[1] - goal[1]))
        phi_after = (-(abs(nxt[0] - goal[0]) + abs(nxt[1] - goal[1]))
                     if nxt is not None else 0.0)   # Phi(goal)=0
        shaped = float(reward) + 0.9 * phi_after - phi_before
        agent.learn(pos, action, shaped, next_position=nxt)
        memory.record(pos, action, float(reward),
                      agent.nt.dopamine.level, np.zeros(4))
        agent.nt.step()                            # dopamine decays
    return steps


def main() -> None:
    world = GridWorldA({"environment": {"grid_size": 20, "max_steps": 300}}, seed=11)
    memory = EpisodicMemory(capacity=20_000)
    value_table = ValueTable(seed=1)
    nt = NeuromodulatorSystem(PARAMS)
    agent = RPEAgent(value_table, nt)
    consolidation = Consolidation(memory, value_table)
    rng = np.random.default_rng(3)

    steps_per_episode = []
    print(f"{'ep':>3} | {'steps':>5} | {'vs ep1':>7}")
    for ep in range(1, 21):
        steps = run_episode(world, agent, memory, rng, ep)
        steps_per_episode.append(steps)
        consolidation.sleep(replay_k=50)
        print(f"{ep:3d} | {steps:5d} | {steps / steps_per_episode[0]:6.0%}")

    improvement = 1.0 - steps_per_episode[-1] / steps_per_episode[0]
    print(f"\nImprovement: {improvement:.0%} (acceptance: > 50%)")
    assert improvement > 0.5, "Month 3 acceptance FAILED"

    with open("docs/benchmarks.md", "a", encoding="utf-8") as fh:
        fh.write(f"\n| Month 3 | RPE+consolidation | ep1: {steps_per_episode[0]} "
                 f"-> ep20: {steps_per_episode[-1]} steps | improvement "
                 f"{improvement:.0%} |\n")
    print("ACCEPTED: maze learned by the brain's own values")


if __name__ == "__main__":
    main()
