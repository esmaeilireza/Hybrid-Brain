"""Week 19 acceptance v2: the satiation curve, isolated.

v1 postmortem: both arms were IDENTICAL and the verdict read 102%
decline with negative values. Two confounds: (1) learn() reinforced
the value table each rep, so RPE shrank naturally -> dopamine shrank
in BOTH arms (that is TD learning, not satiation); (2) the changing
RPE meant attenuate() never saw the same (pos, reward) signature
twice -> satiation never engaged.

v2 isolation: value table frozen (compute_rpe + dopamine_update only,
no reinforce) -> RPE constant at 1.0 -> plain arm MUST be flat
(the control that proves we measure satiation and nothing else).
Metric: per-repetition release delta = dopamine level change caused
by the release itself (no nt.step between reps -> decay excluded).

Expected satiated factors (k=0.5, floor 0.2), computable a priori:
  1.0, 0.667, 0.5, 0.4, 0.333, 0.286, 0.25, 0.222, 0.2, then 0.2 floor
Decline: (1.0 - 0.2) / 1.0 = 80%.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cognition.consolidation import ValueTable
from cognition.reinforcement import RPEAgent
from core.neurotransmitters import NeuromodulatorSystem
from emotions.motivation import Satiation

PARAMS = {
    "simulation": {"dt_ms": 1.0},
    "neurotransmitters": {"dopamine_baseline": 0.1,
                          "dopamine_tau_ms": 100.0},
}
CELL, ACTION, REWARD, REPS = (2, 2), 0, 1.0, 12


def run(with_satiation: bool) -> list[float]:
    agent = RPEAgent(ValueTable(seed=1),
                     NeuromodulatorSystem(PARAMS),
                     satiation=Satiation() if with_satiation else None)
    deltas = []
    for _ in range(REPS):
        before = agent.nt.dopamine.level
        agent.compute_rpe(CELL, ACTION, REWARD)     # no reinforce: rpe fixed
        agent.dopamine_update(pos=CELL)
        deltas.append(round(agent.nt.dopamine.level - before, 4))
    return deltas


def main() -> None:
    plain = run(False)
    sat = run(True)

    print("rep -> release (satiated vs plain):")
    for i, (a, b) in enumerate(zip(sat, plain), 1):
        bar = "#" * max(0, int(a * 40))
        print(f"  {i:2d}: {a:7.4f} {bar:<42.42s} | plain {b:7.4f}")

    # CONTROL (the v1 lesson): the plain arm must be FLAT
    flat = max(plain) - min(plain) < 1e-9
    print(f"\ncontrol (plain arm flat): {'OK' if flat else 'BROKEN'}")
    if not flat:
        print("INVALID: plain arm moved - an unisolated confound, "
              "verdict void")
        sys.exit(1)

    # mechanism check: satiation actually engaged (first factor 1.0,
    # later factors strictly below it, floored)
    engaged = sat[0] == 1.0 and 0 < sat[-1] < sat[1] < sat[0]
    decline = (sat[0] - sat[-1]) / sat[0]
    print(f"release decline across {REPS} repeats: {decline:.0%} "
          f"(expected 80%: floor 0.2)")
    if engaged and decline >= 0.5:
        print("ACCEPTED: satiation attenuates the dopamine response, "
              "mechanism isolated")
        sys.exit(0)
    print("FAILED: satiation mechanism not demonstrated")
    sys.exit(1)


if __name__ == "__main__":
    main()
