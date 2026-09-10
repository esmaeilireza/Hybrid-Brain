"""A/B diagnostic: accept_maze WITHOUT consolidation replay.
If learning stabilizes without sleep, replay is the corruptor."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from cognition.consolidation import ValueTable
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


def run_episode(world, agent, rng, ep: int, max_steps: int = 300) -> int:
    world.reset()
    steps = 0
    while not world.is_episode_done() and steps < max_steps:
        pos = world.agent
        values = agent.value_signals(pos)
        eps = max(0.05, 0.4 * (1.0 - (ep - 1) / 10.0))
        action = (int(rng.integers(0, 4)) if rng.random() < eps
                  else int(np.argmax(values)))
        world.pending_action = action
        world.step(1)
        steps += 1
        reward = world.sensors()[1].read()[0]
        nxt = None if world.is_episode_done() else world.agent
        goal = world.goal
        phi_b = -(abs(pos[0] - goal[0]) + abs(pos[1] - goal[1]))
        phi_a = (-(abs(nxt[0] - goal[0]) + abs(nxt[1] - goal[1]))
                 if nxt is not None else 0.0)
        # shaping scale 0.1: converged path value ~ (0.1*0.9)/(1-0.9)
        # = 0.9, safely inside the [-2,2] clip. Unscaled shaping drove
        # converged values to ~8.9 -> universal clip saturation ->
        # argmax ties -> policy collapsed to action 0 (A/B verified).
        shaped = float(reward) + 0.1 * (0.9 * phi_a - phi_b)
        agent.learn(pos, action, shaped, next_position=nxt)
        agent.nt.step()
    return steps


def main() -> None:
    world = GridWorldA({"environment": {"grid_size": 20, "max_steps": 300}}, seed=11)
    value_table = ValueTable(seed=1)
    agent = RPEAgent(value_table, NeuromodulatorSystem(PARAMS))
    rng = np.random.default_rng(3)

    steps_list = []
    print("A/B: consolidation replay DISABLED")
    print(f"{'ep':>3} | {'steps':>5} | {'vs ep1':>7}")
    for ep in range(1, 21):
        steps = run_episode(world, agent, rng, ep)
        steps_list.append(steps)
        print(f"{ep:3d} | {steps:5d} | {steps / steps_list[0]:6.0%}")

    improvement = 1.0 - steps_list[-1] / steps_list[0]
    mean_last5 = np.mean(steps_list[-5:])
    print(f"\nep20 improvement: {improvement:.0%}")
    print(f"mean of last 5 episodes: {mean_last5:.0f} steps")

    # value inspection near the goal (friend's diagnostic #2)
    for cell in ((18, 19), (19, 18), (10, 10), (0, 0)):
        v = value_table.values_at(cell)
        print(f"V{cell} = {np.round(v, 2)}")


if __name__ == "__main__":
    main()
