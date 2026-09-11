# ADR-020: Month 7 Rescoped - Scientific Validation & Publication

## Status: Accepted | Month 7 open

## Context
External review by a cognitive neuroscience professor (Germany)
and arXiv submission are the next milestones. ADR-006 already cut
4 of 6 superhuman features; the user now directs FULL removal of
the superhuman scope for this phase.

## Decision
1. speed_thinking.py and multi_tasking.py are DEFERRED (not cut:
   chain-of-perspectives remains a valid future experiment).
2. Month 7 = scientific validation month:
   - W25: neuroscience literature mapping (module -> region ->
     citation) as docs/neuroscience_mapping.md
   - W26: ablation study + paper-grade figures (emotion dynamics,
     cocktail party, habit dissociation, somatic avoidance)
   - W27: EEG simulator (FFT over region activity - from old M8 plan,
     pulled forward: highest value for a neuroscience reviewer)
   - W28: arXiv manuscript skeleton + reproducibility package +
     review demo script
3. Tag scheme: Month 7 closes as v0.8.0-scientific-validation
   (v0.7.0 already consumed by Month 6 perception-behavior;
   legacy v0.6.0-pfc-bg collision documented earlier).
