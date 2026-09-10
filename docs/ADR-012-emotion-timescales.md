# ADR-012: Emotion Timescale Hierarchy
# ADR-012: Emotion Timescale Hierarchy

## Status: Accepted (Week 17 closed)

## Context
Dopamine (neurotransmitters.py, Month 3) decays with tau ~100ms. Emotions
(emotion_engine.py, Week 17) use tau 2-30s. Moods (mood.py, Week 20) will
use tau ~hours. Three timescales, one organism.

## Decision
Three distinct timescale classes, each with a physiological justification:

| Class   | tau        | Function                          | Module              |
|---------|------------|-----------------------------------|---------------------|
| Dopamine| ~100ms     | Transient credit assignment (RPE) | neurotransmitters   |
| Emotions| 2-30s      | Behavioral priors on action value | emotion_engine      |
| Moods   | ~hours     | Slow valence baseline shifting    | emotions/mood (W20) |

Why emotions MUST decay slower than dopamine: RPE is a per-step signal -
it must vanish within a cycle or it contaminates the next credit
assignment. Emotions must OUTLAST the triggering episode (seconds) so
they can bias the BG value function across multiple decisions - fear of
region R must persist long enough to suppress re-entry. Moods must
outlast emotions so they integrate across episodes.

## Executable proof
tests/test_emotion.py::test_timescale_hierarchy_fear_outlasts_surprise
(tau_fear=10s vs tau_surprise=2s, same burst, fear > surprise after 10 steps).

## Consequences
- Emotion() takes tau_s as constructor param - hierarchy is explicit.
- Week 20 mood.py must reuse Emotion with tau_s in the 1800-10800s range.
- Integration consumers (Week 18): flashbulb priority reads arousal;
  somatic markers read (valence, arousal) jointly.

