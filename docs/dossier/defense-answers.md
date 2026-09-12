# Defense Dossier - Part 1: Answers with Evidence
> Every answer has three layers: (1) the written answer, (2) the evidence
> pointer (file + measured numbers), (3) the one-sentence spoken version.
> Rule: no claim without a pointer. If a pointer is missing, the claim
> does not go in the spoken version. (ADR-015 discipline.)
> Current suite: 110 passed, 1 skipped (verify with `pytest -q` before the meeting).

---

## Q2 - Multiscale dynamics (analytical; no code needed)

**Answer:** The architecture coordinates three timescales in one system:
neuronal firing at dt=1.0 ms, dopamine kinetics with tau=100 ms, and
emotional dynamics in the 2-30 s band, with mood states persisting for
hours. This hierarchy is not incidental - it mirrors the biological
timescale separation, and each scale is measured, not asserted.

**Evidence:**
- Parameter calibration: ADR-012 (tau_s as constructor param, W17)
- Emotion dynamics measured: sadness 0->0.16 (tau=30s), anger 0->0.30
  (tau=15s, blocked-escape profile) - W24 addendum, emotion_monitor.py
- f-I response measured 10-20 (W21 checklist)
- Known limitation, stated honestly: the gamma-band approximation is
  documented in the fig7 caption

**Spoken version:** "Millisecond neurons, 100ms dopamine, second-scale
emotions, hour-scale moods - all four measured, all calibrated to
ADR-012, and we publish the limitation on gamma."

---

## Q6 - Where is the control condition? (the centerpiece)

**Answer (measured, not argued):** An ablation study
(`benchmarks/ablation_somatic.py`, 3 seeds: 42/43/44, pre-declared
convergence criterion: 5 consecutive episodes with R-visit rate <= 50%
of the ep1-5 baseline; acceptance rule pre-declared: >=10% advantage
AND ON wins >=2/3 seeds) shows somatic markers accelerate avoidance
learning: mean convergence episode 50.7 (ON) vs 60.0 (OFF, never
converged within budget) - a 16% convergence advantage, ON winning
3/3 seeds. The OFF arm faced a HIGHER baseline visit rate in every seed
(14.8/6.8/9.2 vs 7.2/5.4/4.2), i.e. an easier threshold - so the
measured advantage is conservative. This complements Week-18 behavioral
acceptance: ON 0/10 vs OFF 10/10 visits to the punished region, with
zero navigation cost (eval goals 10/10 both arms).

**The discipline story (say this if the professor probes the 2.1x):**
The original draft cited a "2.1x faster" convergence figure. That number
was never measured. We retracted it, built a real ablation with
pre-declared criteria, and reported the honest result including its
limitations. The v1 harness itself failed (greedy-probe baseline on a
zeroed value table collapsed the criterion to an unreachable floor -
both arms hit the 60-episode budget). We root-caused it, redesigned the
baseline to come from organic training behavior, and recorded the
postmortem. The retraction is a feature of the project, not a scar.

**Limitations, stated unprompted:**
- 3 seeds is not statistically powered
- Arms' exploration baselines differ -> not a perfectly paired design
- INCONCLUSIVE was an acceptable outcome by design; we got ACCEPTED
  within pre-declared rules instead

**Evidence:**
- Script: benchmarks/ablation_somatic.py (in repo, commit c746891)
- Full record: docs/benchmarks.md, Week 30 entry
- Independent behavioral data: docs/benchmarks.md, Week 18 (v3 + v4.1)

**Spoken version:** "We caught an unmeasured number, retracted it, and
replaced it with a pre-registered ablation: 16% convergence advantage,
3 of 3 seeds, conservative by design - and the behavioral effect was
already measured in Week 18: zero vs ten visits to the punished region."

---

## Q9 - Falsifiable prediction (proposed experiment)

**Answer:** Beta-augmented threat drive should accelerate habit
crystallization. The beta->habit chain links the W27 EEG finding
(beta threat/rest ratio 1.72, criterion > 1.2) to ADR-019
(habit formation / inertia after 2 bad outcomes / invalidation after 3).

**Measurable form:** track beta-augmented threat drive + habit
crystallization speed (vs rest drive control) as a benchmark - the
falsifiable prediction then has an in-repo test harness pattern to
follow (same discipline as the Q6 ablation).

**Positioning:** in the dossier/paper this appears as a PROPOSED
EXPERIMENT with the sentence: "a minimal in-repo harness for this is
future work." That is a mature academic stance - implementing it would
be a reversal-learning task plus spectral analysis, which is out of
scope for the defense and would dilute focus.

**Evidence:**
- W27 EEG: beta ratio 1.72 > 1.2 (docs/benchmarks.md, Month 7)
- Habit crystallization: demo_day.py Beat 5 (ADR-019: cached True,
  inertia after 2 bad outcomes True, invalidated after 3 True)

**Spoken version:** "Prediction: augmenting threat drive in the beta
band should speed habit crystallization - measurable against the
ADR-019 habit benchmark, with a minimal harness as declared future
work."

---

## Q1 - (PASTE ACTUAL QUESTION TEXT)
**Answer:** [TO FILL]
**Evidence:** [TO FILL]
**Spoken version:** [TO FILL]

## Q3 - (PASTE ACTUAL QUESTION TEXT)
**Answer:** [TO FILL]
**Evidence:** [TO FILL]
**Spoken version:** [TO FILL]

## Q4 - (PASTE ACTUAL QUESTION TEXT)
**Answer:** [TO FILL]
**Evidence:** [TO FILL]
**Spoken version:** [TO FILL]

## Q5 - (PASTE ACTUAL QUESTION TEXT)
**Answer:** [TO FILL]
**Evidence:** [TO FILL]
**Spoken version:** [TO FILL]

## Q7 - (PASTE ACTUAL QUESTION TEXT)
**Answer:** [TO FILL]
**Evidence:** [TO FILL]
**Spoken version:** [TO FILL]

## Q8 - (PASTE ACTUAL QUESTION TEXT)
**Answer:** [TO FILL]
**Evidence:** [TO FILL]
**Spoken version:** [TO FILL]

---

## Evidence index (one table, for quick lookup during the defense)

| Claim | Number | Where |
|-------|--------|-------|
| Perf baseline | 60,770 -> 212,862 neuron-ticks/s | benchmarks.md M1/W4 |
| Rust neurons-only | 2.7x (two independent runs) | benchmarks.md W12/W13 |
| PyO3 combined | 0.20-0.23x (slower; documented, not hidden) | benchmarks.md W15 |
| Maze learning | 100% success, mean 40 steps, 78/80 | benchmarks.md M3 |
| Decision agreement | 89% optimal, 94% System2-on-novel | benchmarks.md M4 |
| Avoidance (behavioral) | ON 0/10 vs OFF 10/10 R-visits, zero nav cost | benchmarks.md W18 v3+v4.1 |
| Satiation | 80% decline = closed form 1/(1+0.5n) | benchmarks.md W19 |
| OCEAN differentiation | 34% goal-rate diff, reproduced | benchmarks.md W20 |
| Cortex6 | gradient L0>L5, all layers in band, 97 passed | benchmarks.md W21 |
| Cocktail party | attended gain 3x, steal verified | benchmarks.md W22 |
| EEG | beta ratio 1.72 > 1.2, gamma limitation documented | benchmarks.md W27 |
| Q6 ablation | 16% advantage, 3/3 seeds, conservative | benchmarks.md W30 |
| Test suite | 110 passed, 1 skipped | pytest -q (current) |


---

# Defense Dossier - Part 1a: Candidate Questions (PREDRAFTED)
> STATUS: Q1, Q3-Q5, Q7, Q8 are CANDIDATE questions - drafted from the
> project's known pressure points (architecture, generalization, memory,
> decision models) because the exact question texts are unavailable.
> They are NOT the professor's actual questions. If the live question
> differs, these drafts supply the raw material - answer by combining
> the closest draft with the evidence index in defense-answers.md.
> (Marked CANDIDATE per ADR-015 spirit: no invented certainty.)

---

## Q1 (CANDIDATE) - Why hybrid? Why not pure spiking or pure RL?

**Answer:** The design thesis is that biological plausibility and
learning performance are not enemies - each pillar is accepted only
with a measurable gate, and each mechanism earns its place by ablation,
not by analogy. The somatic-marker ablation (W30) is the template:
the mechanism is kept because removal measurably hurts (16% convergence
advantage lost, 3/3 seeds), not because Damasio said so.

**Evidence:** Q6 ablation (16%, 3/3 seeds); W18 behavioral 0/10 vs
10/10; W15 honest PyO3 regression (0.2x) - hybrid engineering includes
measuring where the hybrid does NOT pay off.

**Spoken version:** "Every component survived a removal test, not an
appeal to biology - and where the hybrid was slower, we documented it."

---

## Q3 (CANDIDATE) - Memory architecture: how does consolidation work?

**Answer:** Memory is layered: sensory (~500ms), working (Baddeley
7+/-2), episodic (replay of high-reward episodes during the sleep
phase -> consolidation), semantic, and procedural (in basal ganglia -
which is also why habit caching lives there per ADR-019).
Consolidation was UNSUSPENDED per ADR-005 and replay targets the
highest-reward episodes, mirroring hippocampal replay.

**Evidence:** W17-W18 history entries (ADR-005); demo_day Beat 5
(habit cached / inertia / invalidation all True); ADR-019.

**Spoken version:** "Episodic replay of high-reward episodes
consolidates into procedural habit in the basal ganglia - and the
whole chain is unit-tested: formation, inertia, invalidation."

---

## Q4 (CANDIDATE) - Generalization: your results are one grid world - do they transfer?

**Answer (honest):** Partially - and we say exactly where it stops.
The avoidance/somatic result is behavior-level, not world-specific:
it depends on the marker mechanism, not the map. But the OCEAN
benchmark (W20) explicitly showed regime-dependence (bump direction
FLIPPED between 10x10 and 20x20) - we recorded that as a finding, not
hid it. Cross-world transfer (GridWorld-B, VisualWorld) is the
declared open item, and the 20x20 tuning item is open by name.

**Evidence:** benchmarks.md W20 (DIRECTION FLIPPED, OPEN TUNING
ITEM); W22-W24 (VisualWorld, auditory - second environment already
exercised for perception, not yet for RL transfer).

**Spoken version:** "One mechanism-level result I claim generally;
the behavior-level results I scope to their regimes - and the file
where regime-dependence flipped is public."

---

## Q5 (CANDIDATE) - Decision making: System 1 vs System 2 - is this real dual-process or just a router?

**Answer:** It is a familiarity-based router with measured behavior,
not a claimed cognitive implementation: 89% optimal agreement overall,
94% System2-on-novel. We do not claim consciousness or Kaneman-fidelity;
we claim a measured dispatch rule with a measured agreement rate - and
the honest limitation is that agreement was measured on the training
distribution.

**Evidence:** benchmarks.md M4 (89%/94%); ADR-018 (narration observes,
never decides - hard architectural boundary).

**Spoken version:** "It is a measured router, not a philosophical
claim: 89% agreement, 94% when novel - and the LLM narrates but is
architecturally forbidden from deciding."

---

## Q7 (CANDIDATE) - Validation: how do you know the brain model is biologically credible and not just a game AI?

**Answer:** Three layers. (1) Physiological calibration: real values
(tau_m=20ms, v_rest=-65mV, v_threshold=-55mV, dt=1ms), f-I measured
10-20, cortex6 accepted with zero asterisks (97 passed). (2)
Neurotransmitter kinetics measured: dopamine tau=100ms, satiation
matches closed form 1/(1+0.5n) exactly - 80% decline, predicted 80%.
(3) Neural-oscillation evidence: EEG beta threat/rest ratio 1.72 > 1.2,
with the gamma approximation limitation documented in the figure
caption. What we do NOT claim: spike-timing fidelity at HH level
(HH is a stub per roadmap), gamma-band accuracy.

**Evidence:** benchmarks.md W21, W19, W27; ADR-012; fig7 caption.

**Spoken version:** "Calibration is physiological, kinetics match a
closed form exactly, and the EEG beta signature exceeds criterion -
while the gamma limitation and HH stub are documented, not hidden."

---

## Q8 (CANDIDATE) - Engineering: why did the Rust rewrite end up slower, and what did you learn?

**Answer:** Measured honestly: neurons-only Rust was 2.7x faster
(two independent runs), but the combined pipeline via PyO3 is
0.20-0.23x - boundary-crossing overhead dominates. This is recorded,
not hidden, and it taught the standing rule behind every benchmark in
this project: a mechanism claim needs its control arm, and a
performance claim needs its regime. The 2.7x survives where the
boundary is crossed rarely; it dies where crossing is per-tick.

**Evidence:** benchmarks.md W12/W13 (2.7x twice), W15 (0.20-0.23x +
honest note), W30 postmortem (same discipline pattern).

**Spoken version:** "2.7x where the boundary is rare, 0.2x where it
is per-tick - measured, documented, and turned into a rule that now
gates every claim in the project."

---

## Rules for using CANDIDATE drafts live

1. Listen for which draft is closest; do NOT recite - combine.
2. Every number must come from the evidence index; nothing new.
3. If the question has no matching draft: answer with structure
   (claim -> evidence pointer -> limitation) and offer to follow up
   with the exact file and line after the meeting.
4. The retraction story (Q6) is the fallback credibility move for ANY
   question about rigor: we demonstrably retract unmeasured claims.
