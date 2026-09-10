# ADR-005: Consolidation Suspension and Redesign

## Status: Accepted | Month 3

## Context
Week 11 A/B (active/inactive consolidation, identical episode tables and byte-for-byte)
proved that replay did not corrupt online values — but the Week 12 review identified
magnitude domination: replay
(lr * weight * up-to-10x per episode per cell, clip 2.0) is ~10000x larger
than online updates (0.5xRPE per step). Any coexistence
with per-episode replay initialization pushes the table toward a small set
of old entries.

## Decision
1. Consolidation is suspended from the acceptance path until redesign
   (sleep() is a documented no-op; the test is marked xfail with a reason)
2. Redesign schema (Month 4+): weight-capped replay — replay entries
   are capped per (position,action) per sleep hour; replay learning rate
   0.1x online rate; priority refresh based on |RPE| (surprises) not just reward
3. After redesign, the xfail test is re-enabled with the same original assertion

## Consequences
- Month 3 learning path is clean (100% success, 40 steps)
- Consolidation returns as a design commitment instead of a patch
- Reference benchmark: Week 9 acceptance before stabilization: top_rewarded kept correct ranking
