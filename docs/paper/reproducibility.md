# Reproducibility package (W28)

## Environment
python 3.11.9; pip freeze -> requirements-freeze.txt (committed)

## Regenerate every artifact
| Artifact | Command |
|---|---|
| Full test suite (111) | `pytest tests/ -v` |
| Fig1-5 | `python benchmarks/make_figures.py` |
| Fig6-7 (EEG) | `python analysis/eeg_simulator.py` |
| Navigation demo | `python benchmarks/demo_month6.py` |
| Fig5 input | `data/logs/month6_trajectory.csv` (demo output) |

## Seeds
See seed_registry.md. All figures deterministic per seed.

## Known limitations (stated in paper)
Gamma baseline (75Hz drive), simulated EEG proxy, computational-
not-phenomenal affect claims, rat-scale abstractions.
