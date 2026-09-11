"""Week 20 - personality injection, mood timescale, GridWorld-B contract."""
import numpy as np

from emotions.mood import Mood
from emotions.personality import TraitVector
from environment.grid_world_a import GridWorldA
from environment.grid_world_b import GridWorldB


def test_traits_map_to_parameter_ranges() -> None:
    t = TraitVector(openness=0.0, conscientiousness=1.0,
                    agreeableness=0.0, neuroticism=0.0)
    assert t.curiosity_mult == 0.5 and t.epsilon_floor == 0.05
    t2 = TraitVector(openness=1.0, conscientiousness=0.0,
                     agreeableness=1.0, neuroticism=1.0)
    assert t2.curiosity_mult == 3.5 and t2.epsilon_floor == 0.30
    assert t2.fear_tau_scale == 3.0 and t2.neg_rpe_gain == 3.0
    assert t2.marker_gain_mult == 1.5


def test_mood_outlasts_fear_by_orders_of_magnitude() -> None:
    """ADR-012 third rung - computed expectations: after 2 simulated
    seconds of held valence 0.8:
      fear (tau 10s)   -> 0.8*e^(-0.2)  = 0.655  (18% decayed)
      mood (tau 3600s) -> 0.8*(1-e^(-2/3600)) ~ 4.6e-4 (0.06% moved)
    ~300x separation. Snap-to-target 'fixes' are banned: a 3600s
    timescale that jumps instantly has no timescale."""
    from emotions.emotion_engine import Emotion
    fear = Emotion("fear", 0.0, 10.0, 50.0)
    mood = Mood(tau_s=3600.0, dt_ms=50.0)
    fear.burst(0.8)
    for _ in range(40):
        fear.step(0.0)
        mood.step(0.8)
    assert abs(fear.level - 0.8 * float(np.exp(-0.2))) < 0.01
    assert mood.valence < 0.008                      # <1% of the way
    fear_frac = (0.8 - fear.level) / 0.8
    mood_frac = mood.valence / 0.8
    assert fear_frac > 100 * mood_frac               # the hierarchy itself


def test_grid_b_implements_world_protocol() -> None:
    from core.interfaces import World
    w = GridWorldB({"environment": {"fixed_start": [0, 0]}}, seed=1)
    assert isinstance(w, World)
    assert len(w.sensors()) == 2 and len(w.actuators()) == 1


def test_grid_b_reward_only_via_sensor_and_goal_ends() -> None:
    w = GridWorldB({"environment": {"fixed_start": [0, 0]}}, seed=1)
    w.reset()
    w.agent = (w.goal_low[0], w.goal_low[1] - 1)
    w.pending_action = 3                # right -> onto low goal
    w.step(1)
    total = w.sensors()[1].read()[0]
    # computed: goal_low 0.5 - step_penalty 0.01 = 0.49 (friend's read:
    # the THRESHOLD was arithmetically sloppy, not the world design)
    assert total >= 0.45
    assert w.is_episode_done()


def test_grid_b_bumps_register_and_block() -> None:
    w = GridWorldB({"environment": {"fixed_start": [0, 0]}}, seed=1)
    w.reset()
    wall_col = w.size // 2
    w.agent = (0, wall_col - 1)
    before = w.agent
    w.pending_action = 3                # into the wall
    w.step(1)
    assert w.agent == before
    assert w.bumps == 1
    assert w.sensors()[1].read()[0] < 0
