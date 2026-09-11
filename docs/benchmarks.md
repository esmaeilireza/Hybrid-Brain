# Hybrid-Brain - Benchmark Record

| Hardware | Backend | Neurons | Ticks | Throughput | Wall time | Mean rate |
|----------|---------|---------|-------|------------|-----------|-----------|
| i7-6700HQ / 24GB | Python+NumPy+scipy | 10000 | 1000 | 60,770 neuron-ticks/s | 164.55s | 94.5 Hz |
| i7-6700HQ / 24GB | Python+NumPy+scipy | 10000 | 1000 | 74,642 neuron-ticks/s | 133.97s | 94.5 Hz |
| i7-6700HQ / 24GB | Python+NumPy+scipy | 10000 | 1000 | 212,862 neuron-ticks/s | 46.98s | 34.4 Hz |

| Month 3 | RPE + distance-delta shaping + entry-only novelty + fixed start | success 100%, mean 40 steps (training goals 78/80) |

| Week 12-13 | neurons-only A/B | Python 34,641,728 vs Rust 94,381,471 neuron-ticks/s | speedup 2.7x |

| Week 12-13 | neurons-only A/B | Python 28,290,456 vs Rust 77,402,376 neuron-ticks/s | speedup 2.7x |

| Week 14 | neurons+synapses Python-only | 64,989,082 neuron-ticks/s (Rust combined pending PyO3, Week 15) |

| Week 15 | combined via PyO3 | Python 65,557,865 vs Rust 13,184,498 | ratio 0.20x |

| Month 4 | decision agreement | 89% optimal agreement, System2-on-novel 94% |

| Week 15 | combined via PyO3 | Python 57,550,313 vs Rust 13,457,719 | ratio 0.23x |

| Month 4 | decision agreement | 89% optimal agreement, System2-on-novel 94% |

| Month 3 | RPE + distance-delta shaping + entry-only novelty + fixed start | success 100%, mean 40 steps (training goals 78/80) |

| Month 4 | decision agreement | 89% optimal agreement, System2-on-novel 94% |

## Week 18 - Conditioned Avoidance (Month 5 first acceptance)
- benchmarks/accept_avoidance.py v3 - A/B ablation, seed 42, GridWorld-A seed 11
- R = greedy-path corridor cell; threat channel isolated from RL channel
- Train: 77/80 both arms | Eval goals: 10/10 both arms
- R-visits (deterministic eval): somatic ON = 0, OFF = 10
- VERDICT: ACCEPTED - 100% avoidance, zero navigation cost
- Harness lesson (3 versions): v1 false-positive -> competency gate rule;
  v2 confound (punishment in agent.learn for both arms) -> isolated threat
  channel; v3 valid. Acceptance numbers without baseline competence are noise.

## Week 18 final - avoidance with REAL amygdala (v4.1, tag v0.4.2-week18-avoidance)
- R=(0,1) selected by the eval-identical greedy path walker (v4 probe bug fixed)
- Train: 77/80 both arms | Eval goals: 10/10 both arms
- R-visits (deterministic eval): somatic ON = 0, OFF = 10
- VERDICT: ACCEPTED - 100% avoidance, zero navigation cost, amygdala-driven
- Chain complete, no stand-ins: punishment -> amygdala (3-factor, dopamine-
  gated) -> fear -> somatic marker (read-time bias) -> avoidance

## Week 19 - Satiation curve, isolated (ADR-015)
- bench_satiation.py v2: value table frozen, RPE constant 1.0, plain arm
  flat (control OK) vs satiated arm declining 1.0 -> 0.2 floor over 12 reps
- Measured factors: 1.0, 0.667, 0.5, 0.4, 0.333, 0.286, 0.25, 0.222, 0.2
  (floor hit at rep 9) - matches closed form 1/(1+0.5n) exactly
- Decline: 80% (predicted 80%)
- v1 postmortem: first benchmark was a false positive - TD-learning
  shrinkage read as "satiation" (arms identical, verdict 102% with
  negative values). New rule: a mechanism benchmark needs a FLAT
  control arm, or it measures the environment, not the mechanism.
- MotivationSystem (energy>safety>curiosity) + Curiosity (0.5/(1+n/3),
  saturating) unit-verified; behavioral integration in GridWorld-B
  (Week 20, per ADR-007)
