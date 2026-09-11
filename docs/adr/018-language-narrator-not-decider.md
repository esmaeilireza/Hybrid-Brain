# ADR-018: Language as State Narrator, Never Decision Maker

## Status: Accepted | Week 22 (homework gate PASSED)

## Decisions
1. cognition/language.py has READ-ONLY access to brain telemetry.
   No write path to MotorActuator, ActionSpace, or decision layers.
2. Anchor - Anti-annexation (ADR-2): an LLM that acts is an
   untrained foreign brain - outside RPE/plasticity/consolidation;
   nothing teaches it.
3. Anchor - Attribution: 'why this action' must stay answerable
   (System 1/2, somatic markers, needs, OCEAN) - an LLM decision
   replaces it with 'the model predicted that token'.
4. Language = telemetry interface: consumes GlobalWorkspace,
   attention focus, emotions; emits first-person narration
   (cognitive telemetry, like the heatmap but text).
5. LanguageModel Protocol: RuleBasedLanguage (deterministic,
   default, Pi-safe) and OllamaLocal (enriches narration only).
