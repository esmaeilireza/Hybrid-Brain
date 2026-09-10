import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import inspect
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


def make_world():
    return GridWorldA(
        {"environment": {"grid_size": 20, "max_steps": 300}},
        seed=11,
    )


def make_agent():
    return RPEAgent(ValueTable(seed=1), NeuromodulatorSystem(PARAMS))


def manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def print_header(title):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def try_read_reward(world):
    try:
        sensors = world.sensors()
        print(f"sensors() length = {len(sensors)}")
        vals = []
        for i, s in enumerate(sensors):
            try:
                vals.append((i, s.read()))
            except Exception as e:
                vals.append((i, f"<read failed: {e}>"))
        print("sensor reads:", vals)
        try:
            reward = sensors[1].read()[0]
            print("reward from sensors()[1].read()[0] =", reward)
        except Exception as e:
            print("reading reward from sensor[1] failed:", repr(e))
    except Exception as e:
        print("world.sensors() failed:", repr(e))


def check_source_file_markers():
    print_header("1) CHECKING benchmark file content markers")
    p = Path("benchmarks/accept_maze.py")
    print("exists:", p.exists())
    if not p.exists():
        print("ERROR: benchmarks/accept_maze.py does not exist")
        return

    text = p.read_text(encoding="utf-8", errors="replace")
    markers = [
        "dist_b - dist_a",
        "rng.choice(best)",
        "N_TRAIN",
        "training goal hits:",
        "0.1 * (0.9 * phi_a - phi_b)",
        "np.argmax(values)",
    ]
    for m in markers:
        print(f"{m!r}: count={text.count(m)}")

    print("\nFirst 30 lines:")
    for i, line in enumerate(text.splitlines()[:30], start=1):
        print(f"{i:02d}: {line}")


def inspect_world_basics():
    print_header("2) WORLD BASICS")
    world = make_world()
    world.reset()

    print("type(world):", type(world).__name__)
    print("start agent:", getattr(world, "agent", "<missing>"))
    print("goal:", getattr(world, "goal", "<missing>"))
    print("is_episode_done():", world.is_episode_done())
    print("has pending_action:", hasattr(world, "pending_action"))
    print("dir(world) contains step/reset/sensors:",
          all(hasattr(world, x) for x in ["step", "reset", "sensors"]))

    try_read_reward(world)


def test_action_mapping_from(pos):
    print_header(f"3) ACTION MAPPING FROM {pos}")
    for action in range(4):
        world = make_world()
        world.reset()
        world.agent = pos
        before = world.agent
        world.pending_action = action
        try:
            world.step(1)
            after = world.agent
            done = world.is_episode_done()
            reward = None
            try:
                reward = world.sensors()[1].read()[0]
            except Exception as e:
                reward = f"<reward read failed: {e}>"
            print(
                f"action={action} | before={before} -> after={after} "
                f"| moved={after != before} | done={done} | reward={reward}"
            )
        except Exception as e:
            print(f"action={action} failed:", repr(e))


def test_shaping_signs():
    print_header("4) SHAPING SIGN CHECKS")
    world = make_world()
    world.reset()
    goal = world.goal
    print("goal:", goal)

    # examples around start
    samples = [
        ((0, 0), (0, 0), "stay at corner"),
        ((0, 0), (1, 0), "move closer?"),
        ((1, 0), (0, 0), "move back?"),
        ((5, 5), (5, 6), "side step"),
        ((5, 5), (6, 5), "side step 2"),
    ]

    for pos, nxt, label in samples:
        phi_b = -manhattan(pos, goal)
        phi_a = -manhattan(nxt, goal)
        old_shape = 0.1 * (0.9 * phi_a - phi_b)
        new_shape = 0.2 * (manhattan(pos, goal) - manhattan(nxt, goal))
        print(
            f"{label:15s} | pos={pos} nxt={nxt} "
            f"| dist {manhattan(pos, goal)}->{manhattan(nxt, goal)} "
            f"| old_shape={old_shape:+.3f} | new_shape={new_shape:+.3f}"
        )


def test_value_update():
    print_header("5) VALUE UPDATE / LEARN() SANITY CHECK")
    agent = make_agent()
    pos = (0, 0)

    before = np.array(agent.value_signals(pos), dtype=float)
    print("values before:", before)

    # Try to positively reinforce action 3 several times
    for _ in range(10):
        agent.learn(pos, 3, 1.0, next_position=(0, 1))
        agent.nt.step()

    after = np.array(agent.value_signals(pos), dtype=float)
    print("values after rewarding action 3 x10:", after)
    print("delta:", after - before)
    print("argmax before:", int(np.argmax(before)))
    print("argmax after :", int(np.argmax(after)))


def test_tie_break_bias():
    print_header("6) ARGMAX TIE-BREAK BIAS")
    agent = make_agent()
    pos = START
    values = np.array(agent.value_signals(pos), dtype=float)
    print("initial values at START:", values)
    print("np.argmax(values):", int(np.argmax(values)))

    rng = np.random.default_rng(3)
    best = np.flatnonzero(values == values.max())
    choices = [int(rng.choice(best)) for _ in range(20)]
    print("random tie-break choices over best actions:", choices)


def random_rollout(no_learning=False, max_steps=25):
    title = "7) RANDOM / GREEDY ROLLOUT TRACE"
    print_header(title)

    world = make_world()
    agent = make_agent()
    rng = np.random.default_rng(3)
    world.reset()
    world.agent = START

    visits = {}
    goal = world.goal

    for step in range(max_steps):
        pos = world.agent
        values = np.array(agent.value_signals(pos), dtype=float)

        # pure random if no_learning, otherwise epsilon-greedy
        if no_learning:
            action = int(rng.integers(0, 4))
            eps = None
        else:
            eps = 0.5
            action = int(rng.integers(0, 4)) if rng.random() < eps else int(np.argmax(values))

        world.pending_action = action
        world.step(1)

        reward = None
        try:
            reward = float(world.sensors()[1].read()[0])
        except Exception as e:
            reward = f"<reward read failed: {e}>"

        nxt = world.agent
        novelty = 0.5 if nxt not in visits else 0.0
        visits[nxt] = visits.get(nxt, 0) + 1

        phi_b = -manhattan(pos, goal)
        phi_a = -manhattan(nxt, goal)
        old_shape = 0.1 * (0.9 * phi_a - phi_b)
        new_shape = 0.2 * (manhattan(pos, goal) - manhattan(nxt, goal))

        print(
            f"step={step:02d} pos={pos} act={action} -> nxt={nxt} "
            f"| values={np.round(values, 3)} "
            f"| reward={reward} novelty={novelty:+.1f} "
            f"| old_shape={old_shape:+.3f} new_shape={new_shape:+.3f}"
        )

        if not no_learning:
            try:
                agent.learn(pos, action, float(reward) + novelty + old_shape, next_position=nxt)
                agent.nt.step()
            except Exception as e:
                print("learn failed:", repr(e))
                break

        if world.is_episode_done():
            print("episode ended")
            break


def train_debug(episodes=10, max_steps=80):
    print_header("8) SHORT TRAINING DEBUG")
    world = make_world()
    agent = make_agent()
    rng = np.random.default_rng(3)
    visits = {}
    goal_hits = 0

    for ep in range(1, episodes + 1):
        world.reset()
        world.agent = START
        unique = {START}

        for step in range(max_steps):
            pos = world.agent
            values = np.array(agent.value_signals(pos), dtype=float)
            eps = max(0.1, 0.5 * (1.0 - (ep - 1) / 20.0))
            action = int(rng.integers(0, 4)) if rng.random() < eps else int(np.argmax(values))

            world.pending_action = action
            world.step(1)
            nxt = world.agent
            unique.add(nxt)

            reward = float(world.sensors()[1].read()[0])

            phi_b = -manhattan(pos, world.goal)
            phi_a = -manhattan(nxt, world.goal)
            old_shape = 0.1 * (0.9 * phi_a - phi_b)

            novelty = 0.5 if nxt not in visits else 0.0
            visits[nxt] = visits.get(nxt, 0) + 1

            agent.learn(pos, action, reward + novelty + old_shape, next_position=nxt)
            agent.nt.step()

            if world.is_episode_done():
                break

        reached = (world.agent == world.goal)
        goal_hits += int(reached)

        print(
            f"ep={ep:02d} steps={step+1:03d} reached={reached} "
            f"final={world.agent} unique_in_ep={len(unique)} "
            f"start_values={np.round(agent.value_signals(START), 3)}"
        )

    print("goal_hits:", goal_hits, "/", episodes)


def greedy_policy_trace(after_train_episodes=20, train_steps=150, eval_steps=60):
    print_header("9) TRAIN THEN GREEDY TRACE")
    world = make_world()
    agent = make_agent()
    rng = np.random.default_rng(3)
    visits = {}

    # short training
    for ep in range(1, after_train_episodes + 1):
        world.reset()
        world.agent = START
        for _ in range(train_steps):
            pos = world.agent
            values = np.array(agent.value_signals(pos), dtype=float)
            eps = max(0.1, 0.5 * (1.0 - (ep - 1) / 20.0))
            action = int(rng.integers(0, 4)) if rng.random() < eps else int(np.argmax(values))

            world.pending_action = action
            world.step(1)
            nxt = world.agent
            reward = float(world.sensors()[1].read()[0])

            phi_b = -manhattan(pos, world.goal)
            phi_a = -manhattan(nxt, world.goal)
            old_shape = 0.1 * (0.9 * phi_a - phi_b)

            novelty = 0.5 if nxt not in visits else 0.0
            visits[nxt] = visits.get(nxt, 0) + 1

            agent.learn(pos, action, reward + novelty + old_shape, next_position=nxt)
            agent.nt.step()

            if world.is_episode_done():
                break

    # greedy trace
    world.reset()
    world.agent = START
    seen = {START}
    max_row = START[0]
    max_col = START[1]

    print("Greedy trace:")
    for step in range(eval_steps):
        pos = world.agent
        values = np.array(agent.value_signals(pos), dtype=float)
        action = int(np.argmax(values))
        world.pending_action = action
        world.step(1)
        nxt = world.agent

        seen.add(nxt)
        max_row = max(max_row, nxt[0])
        max_col = max(max_col, nxt[1])

        print(
            f"step={step:02d} pos={pos} values={np.round(values, 3)} "
            f"argmax={action} -> {nxt}"
        )

        if world.is_episode_done():
            break

    print("final pos:", world.agent)
    print("goal:", world.goal)
    print("goal reached:", world.agent == world.goal)
    print("unique cells visited:", len(seen))
    print("max row reached:", max_row)
    print("max col reached:", max_col)


def show_accept_maze_source_snippet():
    print_header("10) INSPECT CURRENT train_episode SOURCE")
    try:
        import benchmarks.accept_maze as am
        src = inspect.getsource(am.train_episode)
        print(src)
    except Exception as e:
        print("Could not import benchmarks.accept_maze:", repr(e))


def main():
    check_source_file_markers()
    inspect_world_basics()
    test_action_mapping_from((0, 0))
    test_action_mapping_from((1, 1))
    test_shaping_signs()
    test_value_update()
    test_tie_break_bias()
    random_rollout(no_learning=True, max_steps=20)
    random_rollout(no_learning=False, max_steps=20)
    train_debug(episodes=10, max_steps=80)
    greedy_policy_trace(after_train_episodes=20, train_steps=120, eval_steps=50)
    show_accept_maze_source_snippet()


if __name__ == "__main__":
    main()
