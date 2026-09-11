# ADR-016: Personality as Parameter Injection

## Status: Accepted | Week 20 (homework gate PASSED)

## Decision
Personality traits enter as parameter modifiers over EXISTING
mechanisms (multipliers, floors, thresholds, timescales) - never as
behavioral rules (if neurotic: flinch).

## Arguments (from the gate answer)
1. MODULATION NOT BYPASS: a parameter changes probabilities and
   sensitivities; the perception->valuation->selection pipeline stays
   intact. Behavior stays context-sensitive (the Week 18 marker
   precedent: biasing values produced avoidance statistics WITHOUT
   navigation cost; a rule would have destroyed that result).
2. COMPOSABILITY: rule-traits collide on branch order (openness
   explores, neuroticism retreats - which wins?). Parameter-traits
   combine quantitatively on the same mechanisms.
3. BENCHMARK MEANING: two rule-minds are two programs; two
   parameter-minds are two instances of ONE brain whose behavioral
   distributions differ - which is exactly what Month 5 acceptance
   requires ("measurably different behavior statistics").

## Trait -> parameter map (OCEAN)
  Openness          -> Curiosity.bonus_max multiplier (0.5x-2.5x)
  Conscientiousness -> exploration epsilon floor (0.05-0.30)
  Extraversion      -> social reward weight (placeholder, Month 6)
  Agreeableness     -> somatic-marker gain multiplier (punishment
                       sensitivity)
  Neuroticism       -> fear decay tau scale (1x-3x slower) +
                       negative-RPE reinforcement gain (1x-3x)
