# ADR-019: Habit as Cached Motor Policy (S->A), no online RPE

## Status: Accepted | Week 23 (homework gate PASSED)

## Decisions
1. System 1 (W14) is still an ONLINE value evaluation - each step
   reads values_at(pos).argmax(); sensory noise or a transient RPE
   wobble deflects it. A habit is a CACHED S->A policy: zero value
   lookups, zero BG calls - ballistic open-loop execution.
   (Same house style as Cortex6 _rate and cerebellar learning:
   persistent state replaces per-step evaluation.)
2. Habit forms only when the ValueTable is STABLE at a position
   (low recent-RPE variance) - caching a moving target is wrong.
3. Habit-vs-goal dissociation: under reward devaluation / structure
   change (ADR-011), the habit keeps executing (inertia) until an
   accumulated-error detector (accumulated negative RPE while the
   habit runs) INVALIDATES the cache and returns control to the
   goal-directed pathway.
4. Measurement: habit execution is faster (no decision cycles)
   and persists under devaluation until detector trips.
