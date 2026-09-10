# ��� Hybrid-Brain — Hybrid Brain

A hybrid brain simulator: spiking neural network core (SNN) + cognitive architecture + simulated environment.
Goal: run on desktop, deploy on edge, independent of the cloud.

## Architecture
- `core/` — neural core (LIF, Izhikevich, synapse, STDP) — contract: `core/interfaces.py`
- `environment/` — GridWorld (Level A → B) with Sensor/Actuator contract
- `rust-core/` — hot-loop accelerator (activation: Weeks 12–14, PyO3)
- `visualization/` — 2D heatmap (Month 2) → 3D Point Cloud (Month 6)
- `deploy/pi/` — Raspberry Pi deployment target (ADR-002 — documentation only for now)
- `docs/adr/` — architecture decisions (project memory)

## Reference documents
- Roadmap v3.0 (8-month reference document)
- ADR-001: Environment physics level (A → B)
- ADR-002: Raspberry Pi deferral

## Quick start
```bash
python -m venv .venv && source .venv/Scripts/activate
pip install -r requirements.txt
python benchmarks/bench_network.py   # after Week 4
pytest                              # after Week 2
```

## Progress meter
Every week: one “demonstrable capability” + a 30-second demo — not lines of code.
