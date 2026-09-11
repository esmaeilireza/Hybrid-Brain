# Week 24 - Month 6 integration smoke tests (cheap subset of demo).
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from benchmarks.demo_month6 import Brain6, run_episode, cell_of


def test_loop_runs_and_steps():
    b = Brain6()
    b.world.reset()
    traj, _ = run_episode(b, eps=1.0, learn=False)   # random walk
    assert len(traj) > 0
    assert traj[-1]["dist"] >= 0.0


def test_value_table_gets_entries():
    b = Brain6()
    b.world.reset()
    run_episode(b, eps=0.8, learn=True)
    assert len(b.vt.table) > 3, "no cells learned"
