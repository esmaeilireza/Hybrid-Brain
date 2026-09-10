# ADR-004: Homeostatic Synaptic Scaling

## Status: Accepted

## Context
Week 4 benchmark: mean rate 94.5 Hz (hot regime). STDP with
a_minus > a_plus drives slow weight collapse over long runs
(predicted and analytically grounded). Pure STDP cannot hold the
network at a stable operating point on both short and long timescales.

## Decision
Add multiplicative synaptic scaling (Turrigiano 1998): every N ticks,
rescale each postsynaptic row of the E->E weight matrix so its sum
matches a target from config. Multiplicative only - preserves learned
relative structure. Sparse, O(nnz).

## Consequences
- Stabilizes operating point (target mean rate ~20-30 Hz)
- Learned assemblies from STDP are preserved (relative differences kept)
- One-sided clip [0, w_max] for excitatory weights added simultaneously
  (prevents silent weight-sign flips found in Month 1 review)
