# ADR-007: GridWorld-B Deferred to Month 5 + Evaluation Scope Outcome

## Status: Accepted

## Context
Month 3 proved learning in a fixed-start corridor (100% success,
40 steps). Generalization across random starts requires obstacles + objects
(GridWorld-B, Month 5 per the roadmap).

## Decision
- Month 3 evaluation remains official: fixed start, success rate >= 0.8
- Random-start generalization is only claimed/tested after GridWorld-B
- Scope reasons: 20x20 without obstacles is a pure noise-assignment domain;
  GridWorld-B introduces the real exploration domain for which the novelty
  system (Month 5 preview) is designed

## Consequences
- Month 3 success rate remains clean
- GridWorld-B receives a defined use case for the motivational system
