# ADR-006: Month 7 Scope Decision (Week 12 Checkpoint)

## Status: Accepted

## Context
The project is on schedule (Months 1-3 closed with measured criteria).
The Month 7 superhuman survival rule is conditional.

## Decision
Month 7 is reduced and kept: speed_thinking + multi_tasking
only (both combine from existing Month 4+ THREAD primitives;
no new race condition risk). perfect_memory, pattern_mastery,
predictive_simulation, knowledge_integration were removed.

## Reasons
1. Delivery value: 5 parallel chains, concurrency test coverage — achievable in 1 week
2. Removed features are well covered by existing subsystems
   (perfect_memory ~= episodic store + FAISS; knowledge_integration ~= value table + episodic)
3. Saved time is moved to Month 8 documentation and 3D anatomical layer
   — the public face of the product

## Consequences
- Weeks 25-26: speed_thinking + multi_tasking + concurrency tests
- Weeks 27-28: Month 8 work begins (one-week buffer for shocks)
