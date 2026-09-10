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
