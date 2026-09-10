"""Month 4 headline acceptance: 100 constructed scenarios
(50 familiar, 50 novel). Ground truth constructed by design - this
validates the decision stack implementation (System 1 learning,
System 2 loss-aversion math, routing). Acceptance: agreement > 75%
overall; System 2 invoked on >= 90% of novel scenarios."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from cognition.consolidation import ValueTable
from cognition.decision_making import DecisionMaker

GRID = 20


def neighbors(pos):
    r, c = pos
    return [(max(r - 1, 0), c), (min(r + 1, GRID - 1), c),
            (r, max(c - 1, 0)), (r, min(c + 1, GRID - 1))]


def main() -> None:
    rng = np.random.default_rng(7)
    vt = ValueTable(seed=1)
    dm = DecisionMaker(vt, visits={})

    scenarios = []
    for i in range(100):
        pos = (int(rng.integers(0, GRID)), int(rng.integers(0, GRID)))
        nbrs = neighbors(pos)
        best_a = int(rng.integers(0, 4))
        own = np.zeros(4)
        own[best_a] = float(rng.uniform(0.5, 1.5))
        nv = {}
        for a, nb in enumerate(nbrs):
            v = np.zeros(4)
            if a == best_a:
                v[best_a] = 1.2 + float(rng.uniform(0.3, 1.2))
            else:
                v[a] = 0.2 + float(rng.uniform(0.0, 0.2))
            v = v + rng.uniform(-0.05, 0.05, 4)
            nv[a] = (nb, v)
        scenarios.append((pos, own, nv, best_a, i >= 50))

    agree = 0
    s2_novel = 0
    novel_total = 0
    for pos, own, nv, optimal, is_novel in scenarios:
        vt.table[pos[0] * GRID + pos[1]] = own.copy()
        for a, (nb, v) in nv.items():
            vt.table[nb[0] * GRID + nb[1]] = v.copy()
        if not is_novel:
            dm.visits[pos] = dm.visits.get(pos, 0) + 3

        neighbor_values = {a: vt.values_at(nb) for a, (nb, _) in nv.items()}
        action, system = dm.decide(pos, neighbor_values)

        if action == optimal:
            agree += 1
        if is_novel:
            novel_total += 1
            s2_novel += (system == 2)

    rate = agree / len(scenarios)
    s2_rate = s2_novel / max(novel_total, 1)
    print(f"agreement with utility-optimal: {rate:.0%} "
          f"({agree}/{len(scenarios)}) - acceptance > 75%")
    print(f"System 2 invoked on novel states: {s2_rate:.0%} "
          f"- acceptance >= 90%")
    assert rate > 0.75, "Month 4 headline FAILED (agreement)"
    assert s2_rate >= 0.9, "Month 4 FAILED (router on novel states)"

    with open("docs/benchmarks.md", "a", encoding="utf-8") as fh:
        fh.write(f"\n| Month 4 | decision agreement | {rate:.0%} optimal "
                 f"agreement, System2-on-novel {s2_rate:.0%} |\n")
    print("MONTH 4 ACCEPTANCE PASSED")


if __name__ == "__main__":
    main()
