# Benchmarks Summary (dossier quick-reference)
> Distilled from docs/benchmarks.md. Every number measured.

| # | Capability | Result | Record |
|---|-----------|--------|--------|
| 1 | Perf baseline (10k neurons) | 60,770 -> 212,862 neuron-ticks/s | M1/W4 |
| 2 | Rust neurons-only | 2.7x (2 independent runs) | W12/W13 |
| 3 | Rust combined via PyO3 | 0.20-0.23x (regression, documented) | W15 |
| 4 | Maze navigation | 100%, mean 40 steps, 78/80 | M3 |
| 5 | Decision agreement | 89% optimal, 94% System2-on-novel | M4 |
| 6 | Conditioned avoidance | ON 0/10 vs OFF 10/10, zero nav cost | W18 v3+v4.1 |
| 7 | Satiation mechanism | 80% decline = 1/(1+0.5n), exact | W19 |
| 8 | OCEAN differentiation | 34% learning-speed diff (reproduced) | W20 |
| 9 | Cortex6 acceptance | gradient L0>L5, all in band, zero asterisks | W21 |
| 10 | Cocktail party | attended gain 3x, steal verified | W22 |
| 11 | Emotion dynamics | sadness tau 30s, anger tau 15s, measured | W24 |
| 12 | EEG beta | threat/rest 1.72 > 1.2 (gamma limited) | W27 |
| 13 | Q6 ablation | 16% convergence advantage, 3/3 seeds | W30 |
| - | Test suite | 110 passed, 1 skipped | current |
