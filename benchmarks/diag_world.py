import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from environment.grid_world_a import GridWorldA

world = GridWorldA(
    {"environment": {"grid_size": 20, "max_steps": 300}},
    seed=11,
)

print("attributes before reset:")
print(vars(world))

world.reset()
print("\nafter reset:")
print("agent =", getattr(world, "agent", "<missing>"))
print("goal =", getattr(world, "goal", "<missing>"))
print("done =", world.is_episode_done())
print("vars =", vars(world))

world.agent = (0, 0)
world.pending_action = 0

print("\nafter manual start:")
print("agent =", world.agent)
print("goal =", world.goal)
print("done =", world.is_episode_done())

for action in range(4):
    world.reset()
    world.agent = (0, 0)
    world.pending_action = action

    before = world.agent
    result = world.step(1)
    after = world.agent

    print(
        f"action={action} "
        f"before={before} after={after} "
        f"done={world.is_episode_done()} "
        f"step_result={result!r}"
    )
