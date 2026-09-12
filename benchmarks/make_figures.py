# make_figures.py - paper-grade figures (W26, ADR-020).
# Every figure regenerated from seeded scenarios - reproducible.
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["pdf.fonttype"] = 42
OUT = Path(__file__).resolve().parent.parent / "docs" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

def _save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / name, dpi=300, bbox_inches="tight")
    fig.savefig(OUT / (Path(name).stem + ".pdf"), bbox_inches="tight")
    plt.close(fig)
    print("saved:", name)

def fig1_emotions():
    from emotions.emotion_engine import EmotionEngine
    e = EmotionEngine()
    joy, sad, ang, val = [], [], [], []
    for _ in range(120):
        s = e.step(rpe=-0.6, distance_reduced=False)
        joy.append(s["joy"]); sad.append(s["sadness"])
        ang.append(s["anger"]); val.append(s["valence"])
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 6), sharex=True)
    ax1.plot(joy, label="joy"); ax1.plot(sad, label="sadness")
    ax1.plot(ang, label="anger")
    ax1.set_ylabel("level"); ax1.legend()
    ax1.set_title("Emotion dynamics under sustained failure")
    ax2.plot(val, color="k"); ax2.set_ylabel("valence")
    ax2.set_xlabel("cognitive cycle")
    _save(fig, "fig1_emotion_dynamics.png")

def fig2_cocktail():
    from cognition.attention import AttentionSystem
    att = AttentionSystem(n_streams=2, seed=0)
    gains, attended = [], []
    for t in range(60):
        v = 5.0
        au = 1.0 if t < 30 else 9.0
        g = att.step([v, au])
        gains.append(g[1]); attended.append(att.attended_idx)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 6), sharex=True)
    ax1.plot([5.0] * 30 + [5.0] * 30, label="visual salience")
    ax1.plot([1.0] * 30 + [9.0] * 30, label="auditory salience")
    ax1.axvline(30, ls="--", c="gray")
    ax1.set_ylabel("salience"); ax1.legend()
    ax2.plot(gains, label="auditory gain")
    ax2.plot(attended, label="attended idx")
    ax2.set_xlabel("cognitive cycle"); ax2.legend()
    ax2.set_title("Cocktail party: loud event steals attention")
    _save(fig, "fig2_cocktail_party.png")

def fig3_habit():
    from behavior.habits import Habits
    h = Habits(invalidate_after=3)
    for _ in range(10):
        h.observe((2, 2), 0, rpe=0.5)
    control, rpes = [], []
    for step in range(20):
        rpe = 0.5 if step < 8 else -0.5
        h.report_outcome((2, 2), rpe)
        control.append(1 if h.get((2, 2)) is not None else 0)
        rpes.append(rpe)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 5), sharex=True)
    ax1.plot(rpes); ax1.set_ylabel("RPE")
    ax1.axvline(8, ls="--", c="gray", label="devaluation onset")
    ax1.legend()
    ax2.plot(control); ax2.set_ylabel("habit controls (1=yes)")
    ax2.set_xlabel("step")
    ax2.set_title("Habit inertia, then invalidation (ADR-019)")
    _save(fig, "fig3_habit_dissociation.png")

def fig4_somatic():
    from emotions.somatic import SomaticMarkerMap
    m = SomaticMarkerMap()
    accrual, extinction = [], []
    for _ in range(20):
        m.record((3, 3), punishment=-0.8, arousal=0.9)
        accrual.append(m.penalty((3, 3)))
    for _ in range(20):
        m.note_safe_visit((3, 3))
        extinction.append(m.penalty((3, 3)))
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(range(20), accrual, label="accrual")
    ax.plot(range(20, 40), extinction, label="extinction")
    ax.axhline(0, c="k", lw=0.5)
    ax.set_xlabel("event"); ax.set_ylabel("marker penalty")
    ax.set_title("Somatic marker: accrual, extinction")
    ax.legend()
    _save(fig, "fig4_somatic_marker.png")

def fig5_navigation():
    csv = Path(__file__).resolve().parent.parent / "data" / "logs" \
         / "month6_trajectory.csv"
    if not csv.exists():
        print("SKIP fig5: run demo_month6 first")
        return
    rows = [l.split(",") for l in csv.read_text().splitlines()[1:] if l]
    xs = [float(r[1]) for r in rows]; ys = [float(r[2]) for r in rows]
    da = [float(r[5]) for r in rows]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.plot(xs, ys, "-o", ms=2); ax1.plot(3.0, 1.0, "r*", ms=14)
    ax1.set_title("trajectory (visual-motor loop)")
    ax2.plot(da); ax2.set_title("dopamine")
    _save(fig, "fig5_navigation.png")

if __name__ == "__main__":
    fig1_emotions()
    fig2_cocktail()
    fig3_habit()
    fig4_somatic()
    fig5_navigation()
    print("ALL FIGURES ->", OUT)
