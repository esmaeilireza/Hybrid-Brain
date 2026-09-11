"""
Hybrid-Brain Interactive History Dashboard
Generates a premium, nested drill-down HTML report.
Month -> Week -> Narrative + Commits + Tags
"""
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "logs" / "history.html"
OUT.parent.mkdir(parents=True, exist_ok=True)

# --- DATA MODELS ---
@dataclass
class WeekInfo:
    title: str
    narrative: str
    punchline: str

MONTHS = {
    1: "Neural Core + Environment", 2: "Brain Regions",
    3: "Memory & Learning", 4: "Cognition + Rust",
    5: "Emotions & Personality", 6: "Perception & Behavior",
    7: "Superhuman (conditional)", 8: "Polish & Product",
}

# Curated Narratives (Weeks 1-18)
WEEKS_DATA = {
    1: WeekInfo("Infrastructure & phase zero", "Folder tree, core/interfaces.py: the immutable Sensor/Actuator/World contract. brain_config.yaml with REAL physiology. BrainClock, git+pre-commit. ADR-001/002.", "First line of the contract that every later module obeyed."),
    2: WeekInfo("Neurons, vectorized from line one", "LIF in NumPy, Izhikevich 20 rhythms, HH deliberately a stub. f-I curve validated against reference. Sub-tick interpolation -> ADR-003.", "Acceptance: spiking above rheobase, refractory respected, f-I monotonic."),
    3: WeekInfo("Synapses + the environment arrives early", "Excitatory/inhibitory synapses on scipy.sparse, delays, neurotransmitters. GridWorld-A built NOW, fixing roadmap error #6.", "Contract test: reward flows ONLY through Sensors."),
    4: WeekInfo("Network + the official metric", "NeuralNetwork + STDP. benchmarks/bench_network.py: neuron-ticks/s becomes the project's speed metric. 10k-neuron self-organizing net.", "MONTH 1 ACCEPTED."),
    5: WeekInfo("Brain regions I", "Thalamus (deterministic relay), hippocampus (place cells), amygdala (three-factor fear conditioning), hypothalamus (homeostasis).", "Week 6 test suite born (test_week6.py still green today)."),
    6: WeekInfo("Brain regions II + EI balance", "PFC (working memory), basal ganglia (Go/No-Go), cerebellum stub. Homeostatic synaptic scaling. EI-balance benchmark.", "The attractor-instability lesson: sweep harnesses become standard."),
    7: WeekInfo("Connectome + version 0.1", "core/connectome.py wires regions; signal flow visible in heatmaps. Visualization from Month 2, not Month 8.", "VERSION 0.1 tagged - the first runnable brain."),
    8: WeekInfo("Cortex (single layer, honestly)", "Single-layer cortex; six layers deferred to Month 6. 2D debug heatmap tooling matures.", "Discipline: deferral WITH an ADR, not silent scope-cut."),
    9: WeekInfo("Memory systems", "Sensory (~500ms), working (Baddeley 7+/-2), episodic (reward-ranked), semantic, procedural lives IN the basal ganglia.", "ChromaDB pending; dict-backed with stable interface."),
    10: WeekInfo("Reinforcement learning (RPE)", "RPE dopamine pathway (Schultz 1997). TD bootstrap gamma=0.9 MANDATORY. Reward enters via RewardSensor only (ADR-2).", "MONTH 3 TAG: v0.3.0-learning - maze learned: 78/80 train goals."),
    11: WeekInfo("Consolidation v1 - honest suspension", "Replay built, then Week-11 A/B proved replay magnitude dominated online learning. Consolidation SUSPENDED with ADR-005 redesign spec.", "Suspended-honestly beats broken-quietly. (Redeemed in W18.)"),
    12: WeekInfo("Rust begins + Week-12 checkpoint", "neurons.rs - bit-exact contract. Checkpoint ADRs: 005, 006, 007.", "The checkpoint rule: decide about Month 7 NOW, not in Month 7."),
    13: WeekInfo("Attention + Global Workspace", "Cocktail-party gain modulation, dwell/hysteresis anti-flicker, 7+/-2 priority broadcast queue.", "Week 13 closed: 52 passed, 1 xfail."),
    14: WeekInfo("Decision making + synapses.rs", "System 1/2 Kahneman with familiarity router, loss aversion lambda=2.0. synapses.rs bit-exact. ADR-008 Option A.", "Archaeology: second-session crate recovered, ADR-013 written."),
    15: WeekInfo("Month 4 headline numbers", "Decision agreement with utility-optimal: 89%. System 2 invoked on novel states: 94%. PyO3 bindings importable.", "MONTH 4 ACCEPTED -> tag v0.4.0-cognition."),
    16: WeekInfo("Language adapters (Month 4 close)", "LanguageModel Protocol: RuleBasedLanguage default. Broca/Wernicke as template-selection + semantic-slot stages.", "Adapter pattern: interface stability over storage tech."),
    17: WeekInfo("The emotion engine", "Six Ekman emotions, Valence-Arousal map, timescale hierarchy -> ADR-012. Emotion takes tau_s as a constructor param.", "Homework gate PASSED: why emotions outlast dopamine. Tag v0.4.1."),
    18: WeekInfo("Integration: the avoidance chain", "Somatic markers (Damasio). Consolidation UNSUSPENDED per ADR-005. A/B ablation: ON 0/10 vs OFF 10/10 visits to R.", "ACCEPTANCE with the REAL amygdala. TAG: v0.4.2-week18-avoidance."),
}

TAG_HINT = {
    "v0.3.0-learning": 10, "v0.3.1-learning": 10,
    "v0.4.0-cognition": 15, "v0.4.1-week17-emotion": 17,
    "v0.4.2-week18-avoidance": 18
}

# --- GIT HELPERS ---
def run_git(*args):
    try:
        res = subprocess.run(['git'] + list(args), capture_output=True, text=True, check=True, cwd=ROOT)
        return res.stdout.strip()
    except subprocess.CalledProcessError:
        return ""
    except FileNotFoundError:
        print("Error: git not found in PATH.")
        sys.exit(1)

# --- HTML GENERATION ---
def generate_html():
    raw_log = run_git("log", "--pretty=format:%h|%ad|%s", "--date=format:%Y-%m-%d")
    tags_raw = run_git("tag", "--sort=creatordate")
    
    rows = [line.split("|", 2) for line in raw_log.splitlines() if line.count("|") == 2]
    tags = tags_raw.splitlines() if tags_raw else []

    if not rows:
        print("No commits found in git history.")
        return

    # Robust Week Mapping: Based on 7-day intervals from the very first commit
    dates = [datetime.strptime(r[1], "%Y-%m-%d") for r in rows]
    first_date = min(dates)
    
    weeks_commits = {}
    for h, d_str, s in rows:
        d = datetime.strptime(d_str, "%Y-%m-%d")
        project_week = ((d - first_date).days // 7) + 1
        weeks_commits.setdefault(project_week, []).append((h, s, d_str))

    tags_by_week = {}
    for t in tags:
        if (hw := TAG_HINT.get(t)):
            tags_by_week.setdefault(hw, []).append(t)

    max_week = max(max(weeks_commits.keys()), max(WEEKS_DATA.keys())) if weeks_commits else max(WEEKS_DATA.keys())
    
    # Build HTML
    months_html = []
    for month in range(1, 9):
        wk_cards = []
        month_has_data = False
        
        for wnum in range((month - 1) * 4 + 1, month * 4 + 1):
            if wnum > max_week and wnum not in WEEKS_DATA:
                break
                
            info = WEEKS_DATA.get(wnum, WeekInfo(f"Week {wnum}", "No narrative recorded yet.", "Pending."))
            commits = weeks_commits.get(wnum, [])
            if commits: month_has_data = True
            
            # Commits HTML
            comm_html = "".join(
                f'<div class="commit"><code>{h}</code> {s} <span class="date">{d}</span></div>'
                for h, s, d in reversed(commits)
            )
            
            # Tags HTML
            tag_html = "".join(f'<span class="tag">{t}</span>' for t in tags_by_week.get(wnum, []))
            
            wk_cards.append(f"""
            <details class="week">
                <summary>
                    <span class="w-num">W{wnum:02d}</span>
                    <span class="w-title">{info.title}</span>
                    <span class="w-meta">{len(commits)} commits</span>
                    {tag_html}
                </summary>
                <div class="w-body">
                    <p class="narrative">{info.narrative}</p>
                    <p class="punchline">🎯 {info.punchline}</p>
                    {f'<details class="commits"><summary>View all commits ({len(commits)})</summary><div class="c-list">{comm_html}</div></details>' if commits else ''}
                </div>
            </details>""")
            
        state_icon = "✅" if month_has_data else "⏳"
        months_html.append(f"""
        <details class="month" {'open' if month in (4, 5) else ''}>
            <summary>
                <span class="m-badge">M{month}</span>
                <span class="m-title">{MONTHS[month]}</span>
                <span class="m-state">{state_icon}</span>
            </summary>
            <div class="m-body">{''.join(wk_cards)}</div>
        </details>""")

    html = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<title>Hybrid-Brain | Project Dashboard</title>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
<style>
  :root {{
    --bg: #0a0a0f; --surface: #12121a; --surface-hover: #1a1a24;
    --border: #2a2a35; --text: #e4e4e7; --text-dim: #8b8b9e;
    --accent: #3b82f6; --accent-glow: rgba(59, 130, 246, 0.15);
    --success: #10b981; --pink: #f472b6;
  }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    background: var(--bg); color: var(--text);
    font-family: 'Inter', system-ui, sans-serif; line-height: 1.6;
    padding: 40px 20px;
  }}
  .dashboard {{ max-width: 960px; margin: 0 auto; }}
  .header {{ text-align: center; margin-bottom: 50px; }}
  .header h1 {{
    font-size: 2.5rem; font-weight: 700; margin-bottom: 8px;
    background: linear-gradient(135deg, #3b82f6, #8b5cf6);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
  }}
  .header p {{ color: var(--text-dim); font-size: 0.95rem; }}
  
  /* Month Accordion */
  .month {{
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 16px; margin-bottom: 16px; overflow: hidden;
    transition: all 0.3s ease;
  }}
  .month[open] {{ box-shadow: 0 0 30px var(--accent-glow); border-color: var(--accent); }}
  .month > summary {{
    list-style: none; cursor: pointer; display: flex; align-items: center;
    gap: 16px; padding: 20px 24px; font-size: 1.1rem; font-weight: 600;
  }}
  .month > summary::-webkit-details-marker {{ display: none; }}
  .m-badge {{
    background: var(--accent); color: #fff; font-size: 0.8rem; font-weight: 700;
    padding: 4px 12px; border-radius: 8px; letter-spacing: 0.5px;
  }}
  .m-title {{ flex: 1; }}
  .m-state {{ font-size: 1.2rem; }}
  
  /* Week Accordion */
  .m-body {{ padding: 10px 20px 20px; background: #0d0d14; }}
  .week {{
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 12px; margin-bottom: 12px;
  }}
  .week > summary {{
    list-style: none; cursor: pointer; display: flex; align-items: center;
    gap: 12px; padding: 14px 18px; flex-wrap: wrap;
  }}
  .week > summary::-webkit-details-marker {{ display: none; }}
  .week > summary:hover {{ background: var(--surface-hover); }}
  .w-num {{
    font-family: monospace; background: #2a2a35; color: var(--accent);
    padding: 2px 8px; border-radius: 6px; font-size: 0.85rem; font-weight: 600;
  }}
  .w-title {{ font-weight: 500; flex: 1; }}
  .w-meta {{ color: var(--text-dim); font-size: 0.8rem; }}
  .tag {{
    background: rgba(59, 130, 246, 0.15); color: var(--accent); border: 1px solid rgba(59, 130, 246, 0.3);
    padding: 2px 10px; border-radius: 20px; font-size: 0.75rem; font-family: monospace;
  }}
  
  /* Week Body Content */
  .w-body {{ padding: 16px 24px 24px; border-top: 1px dashed var(--border); }}
  .narrative {{ color: var(--text-dim); margin-bottom: 16px; font-size: 0.95rem; }}
  .punchline {{
    color: var(--success); font-weight: 500; font-size: 0.95rem;
    padding-left: 12px; border-left: 3px solid var(--success); margin-bottom: 16px;
  }}
  
  /* Commits */
  .commits {{ margin-top: 12px; }}
  .commits summary {{ cursor: pointer; color: var(--text-dim); font-size: 0.85rem; font-weight: 500; }}
  .c-list {{ margin-top: 10px; max-height: 200px; overflow-y: auto; }}
  .commit {{
    font-size: 0.8rem; color: var(--text-dim); padding: 6px 0;
    border-bottom: 1px solid #1a1a24; display: flex; gap: 8px;
  }}
  .commit code {{ color: var(--pink); font-family: monospace; }}
  .commit .date {{ margin-left: auto; opacity: 0.6; font-size: 0.75rem; }}
</style></head><body>
<div class="dashboard">
  <div class="header">
    <h1>🧠 Hybrid-Brain Architecture</h1>
    <p>Interactive Project History • Generated {datetime.now():%Y-%m-%d %H:%M}</p>
  </div>
  {''.join(months_html)}
</div>
</body></html>"""

    OUT.write_text(html, encoding="utf-8")
    print(f"✅ Dashboard generated: {OUT.relative_to(ROOT)}")

if __name__ == "__main__":
    generate_html()