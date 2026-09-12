# Ablation study (Q6, ADR-014): somatic markers ON vs OFF.
# Baseline: mean R-visits of first 5 training episodes (organic behavior).
# Convergence: first episode (after ep 5) where the last 5 episodes'
# max R-visits falls below 50% of that baseline.
# Verdict rule (pre-declared): ACCEPTED only if advantage >= 10%
# AND somatic ON wins (equal or faster) in at least 2 of 3 seeds.
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
from cognition.consolidation import ValueTable
from cognition.decision_making import DecisionMaker
from cognition.reinforcement import RPEAgent
from core.neurotransmitters import NeuromodulatorSystem
from emotions.emotion_engine import EmotionEngine
from emotions.somatic import SomaticMarkerMap
from environment.grid_world_a import GridWorldA

PARAMS = {
    "simulation": {"dt_ms": 1.0},
    "neuron": {"tau_m_ms": 20.0, "v_rest_mV": -65.0,
               "v_threshold_mV": -55.0, "v_reset_mV": -70.0,
               "refractory_ms": 2.0},
    "synapse": {"tau_excite_ms": 5.0, "connection_density": 0.05,
                "excitatory_ratio": 0.8, "delay_min_ms": 1.0,
                "w_exc": 1.5, "w_inh": 3.0},
}
WORLD_CFG = {"environment": {"grid_size": 10, "max_steps": 200}}
SEEDS = (42, 43, 44)
MOVES = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}
PUNISH = -1.0
N_TRAIN = 60

def neighbor(pos, a):
    d = MOVES[a]
    return (pos[0] + d[0], pos[1] + d[1])

def run_arm(somatic_on: bool, seed: int):
    """Train one arm. Returns (convergence_episode, baseline, visits)."""
    vt = ValueTable(seed=1)
    agent = RPEAgent(vt, NeuromodulatorSystem(PARAMS))
    visits, r_visits = {}, {}
    markers = SomaticMarkerMap() if somatic_on else None
    dm = DecisionMaker(vt, visits=visits, somatic=markers)
    eng = EmotionEngine()
    world = GridWorldA(WORLD_CFG, seed=11)
    rng = np.random.default_rng(seed)

    conv_episode = None
    window = []
    base = 1.0

    for ep in range(1, N_TRAIN + 1):
        world.reset()
        world.agent = (0, 0)
        steps, eps = 0, max(0.15, 0.6 * (1.0 - (ep - 1) / 30.0))
        r_here = 0

        while not world.is_episode_done() and steps < world.max_steps:
            pos = world.agent
            nb_pos = {a_: neighbor(pos, a_) for a_ in range(4)}
            nb_vals = {a_: np.array(agent.value_signals(nb_pos[a_]),
                                    dtype=float) for a_ in range(4)}
            action, _ = dm.decide(pos, nb_vals, nb_pos)
            if rng.random() < eps:
                action = int(rng.integers(0, 4))
            world.pending_action = action
            world.step(1)
            steps += 1
            nxt = world.agent
            rw = float(world.sensors()[1].read()[0])
            punished = (nxt == (0, 1))
            total = rw + 0.2 * ((abs(pos[0]) + abs(pos[1]))
                                - (abs(nxt[0]) + abs(nxt[1]))) \
                    + (PUNISH if punished else 0.0)
            rpe = agent.learn(pos, action, total, next_position=nxt)
            agent.nt.step()
            state = eng.step(rpe=rpe,
                             amygdala_activation=0.9 if punished else 0.0)
            if markers is not None:
                if punished:
                    markers.record(nxt, punishment=rpe,
                                   arousal=state["arousal"])
                else:
                    markers.note_safe_visit(nxt)
            if punished:
                r_here += 1

        r_visits[ep] = r_here
        window.append(r_here)

        if ep == 5:
            raw = float(np.mean([r_visits[e] for e in range(1, 6)]))
            base = max(raw, 1.0)
            tag = "ON " if somatic_on else "OFF"
            floor_note = "" if raw >= 1.0 else "  (floored: raw < 1.0)"
            print(f"  [Seed {seed} | Somatic {tag}] "
                  f"Baseline ep1-5: raw {raw:.2f}, used {base:.2f}"
                  f"{floor_note}")

        if conv_episode is None and ep > 5 and \
                max(window[-5:]) <= 0.5 * base:
            conv_episode = ep
        if conv_episode is not None and ep > conv_episode + 3:
            break

    res = conv_episode or N_TRAIN
    tag = "ON " if somatic_on else "OFF"
    print(f"  --> [Seed {seed} | Somatic {tag}] "
          f"Result: {res}, R-visits: "
          f"{[r_visits[e] for e in range(1, 11)]} (ep1-10)")
    return res, base, r_visits

def main():
    print("=== Q6 Ablation: Somatic Markers ON vs OFF (3 seeds) ===")
    on_times, off_times, on_bases, off_bases = [], [], [], []
    for seed in SEEDS:
        res, b, _ = run_arm(True, seed)
        on_times.append(res)
        on_bases.append(b)
        res, b, _ = run_arm(False, seed)
        off_times.append(res)
        off_bases.append(b)

    print("\n--- Summary ---")
    print("somatic ON  convergence episodes:", on_times)
    print("somatic OFF convergence episodes:", off_times)
    print("baselines used - ON:", on_bases, " OFF:", off_bases)

    m_on, m_off = np.mean(on_times), np.mean(off_times)
    adv = (m_off - m_on) / m_off if m_off > 0 else 0.0
    on_wins = sum(1 for o, f in zip(on_times, off_times) if o <= f)
    print(f"mean ON {m_on:.1f} vs OFF {m_off:.1f} -> "
          f"advantage {adv:.0%}; ON wins in {on_wins}/3 seeds")

    floored = any(b <= 1.0 and b == 1.0 for b in on_bases + off_bases)
    if floored:
        print("NOTE: some baselines were floored at 1.0 - "
              "criterion may be stricter than the effect size. "
              "INCONCLUSIVE here is a criterion limit, not proof of no effect.")

    if adv >= 0.10 and on_wins >= 2:
        print("RESULT: ACCEPTED - somatic markers accelerate "
              "avoidance learning (advantage "
              f"{adv:.0%}, {on_wins}/3 seeds)")
    elif adv > 0.0 and on_wins >= 2:
        print("RESULT: TREND ONLY - faster convergence observed but "
              f"below the 10% pre-declared threshold ({adv:.0%}). "
              "Report as 'trend toward faster convergence, "
              "3 seeds, not statistically powered' + Week-18 data.")
    else:
        print("RESULT: INCONCLUSIVE - cite Week-18 behavioral "
              "avoidance acceptance instead (0/10 vs 10/10 visits "
              "to punished region).")

if __name__ == "__main__":
    main()
