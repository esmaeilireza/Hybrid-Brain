# demo_day.py - 10-minute scripted academic demo (W30, ADR-021).
# Six beats, each a PROVEN module re-run deterministically (seeds).
# No new mechanisms. Run: python tools/demo_day.py
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np


def banner(n, title):
    print()
    print("=" * 62)
    print(f"BEAT {n}/6 - {title}")
    print("=" * 62)


def beat1_cortex():
    from regions.cortex6 import Cortex6
    PARAMS = {
        "simulation": {"dt_ms": 1.0},
        "neuron": {"tau_m_ms": 20.0, "v_rest_mV": -65.0,
                   "v_threshold_mV": -55.0, "v_reset_mV": -70.0,
                   "refractory_ms": 2.0},
        "synapse": {"tau_excite_ms": 5.0, "connection_density": 0.05,
                    "excitatory_ratio": 0.8, "delay_min_ms": 1.0,
                    "w_exc": 1.5, "w_inh": 3.0},
    }
    c = Cortex6(240, PARAMS, seed=1)
    for _ in range(50):
        c.step(np.full(c.n_in, 30.0))
    totals = np.zeros(6)
    for _ in range(200):
        counts = c.step(np.full(c.n_in, 30.0))
        totals += np.array(counts, dtype=float)
    sizes = np.array([ly.n_exc for ly in c.layers], dtype=float)
    hz = totals / 200.0 / sizes * 1000.0
    print("six-layer per-neuron rates:", np.round(hz, 1))
    print("all in 2-120 Hz band:", bool(all(2 < h < 120 for h in hz)))


def beat2_emotions():
    from emotions.emotion_engine import EmotionEngine
    e = EmotionEngine()
    for _ in range(120):
        s = e.step(rpe=-0.6, distance_reduced=False)
    print("after sustained failure: sadness", round(s["sadness"], 2),
          "| anger", round(s["anger"], 2), "| valence",
          round(s["valence"], 2))


def beat3_cocktail():
    from cognition.attention import AttentionSystem
    att = AttentionSystem(n_streams=2, seed=0)
    att.step([5.0, 1.0])
    for _ in range(4):
        att.step([5.0, 1.0])
    att.step([5.0, 8.0])
    print("loud auditory steals attention:",
          bool(att.attended_idx == 1), "(gain 3x)")


def beat4_somatic():
    from emotions.somatic import SomaticMarkerMap
    m = SomaticMarkerMap()
    for _ in range(20):
        m.record((3, 3), punishment=-0.8, arousal=0.9)
    print("accrued marker at punished cell:", round(m.penalty((3, 3)), 2),
          "(bounded -0.5)")


def beat5_habits():
    from behavior.habits import Habits
    h = Habits(invalidate_after=3)
    for _ in range(10):
        h.observe((2, 2), 0, rpe=0.5)
    print("habit cached:", h.get((2, 2)) == 0)
    for _ in range(2):
        h.report_outcome((2, 2), rpe=-0.5)
    print("inertia after 2 bad outcomes:", h.get((2, 2)) == 0)
    h.report_outcome((2, 2), rpe=-0.5)
    print("invalidated after 3:", h.get((2, 2)) is None)


def beat6_eeeg():
    from analysis.eeg_simulator import (run_scenario, band_power,
                                        EEGSimulator, BANDS)
    sim = EEGSimulator(seed=1)
    bp_r = band_power(run_scenario(EEGSimulator(seed=1), False, 2000))
    bp_t = band_power(run_scenario(sim, True, 2000))
    r = bp_t["beta"] / bp_r["beta"]
    print("beta threat/rest ratio:", round(r, 2),
          "(criterion > 1.2; gamma limitation documented)")


def main():
    print("HYBRID-BRAIN - ACADEMIC DEMO DAY (6 beats, seeded)")
    for i, (t, fn) in enumerate([
            ("six-layer cortex", beat1_cortex),
            ("emotion dynamics", beat2_emotions),
            ("cocktail party attention", beat3_cocktail),
            ("somatic markers", beat4_somatic),
            ("habit vs goal (ADR-019)", beat5_habits),
            ("EEG spectrum (W27)", beat6_eeeg)], 1):
        banner(i, t)
        fn()
    print()
    print("=" * 62)
    print("DEMO COMPLETE - all six beats ran. Figures in")
    print("docs/figures, numbers in docs/benchmarks.md.")

if __name__ == "__main__":
    main()
