# ADR-001: Choosing the physics level of the GridWorld environment

## Status: ✅ Accepted
## Date: 2026-02

## Context
The Hybrid-Brain project needs an environment to train a neural agent.
Three possible physics levels: A (simple grid), B (movable objects/obstacles), C (PyBullet).

## Decision
Gradual path **A → B**. Level C is outside the current scope.

## Reasons
1. Hardware fit: GTX 960M (4GB VRAM) with PyBullet + STDP simultaneously
   would become a severe bottleneck.
2. Alignment with learning goals: Months 1–4 focus on the neural core.
3. Edge deployment capability: Pi makes Level C impractical.
4. Time risk: Level C costs at least 3 extra months.

## Consequences
- ✅ Faster start (Level A: Week 3, ~150 lines)
- ✅ Smooth visualization on current hardware
- ⚠️ Limited transfer learning to a real robot (acceptable for now)
- ℹ️ Revisit: Month 7, after Level B is complete
