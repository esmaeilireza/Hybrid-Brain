# Neuroscience Mapping - module -> region -> literature (W25)

| Module | Region / phenomenon | Reference |
|---|---|---|
| core/neuron.py, synapse.py | LIF spiking + E/I synapse | validated vs Brian2 (W2) |
| core/neurotransmitters.py | Dopaminergic RPE | Schultz, Dayan & Montague (1997) Science 275:1593 |
| regions/hippocampus.py | Place cells | O'Keefe & Nadel (1978) Cognitive Map |
| regions/amygdala.py | Threat conditioning | LeDoux (2000) Annu Rev Neurosci 23:155 |
| regions/cortex6.py | Layered cortex (6L) | Cortical layer conventions, sweep-tuned (W21) |
| regions/prefrontal.py | Working memory (delay) | Goldman-Rakic (1995) |
| regions/basal_ganglia.py | Go/No-Go selection | Gurney/Redgrave/Prescott selection model |
| cognition/attention.py | Cocktail Party, gating | Posner & Petersen (1990) Annu Rev Neurosci 13:25 |
| cognition/memory.py | WM capacity 7+-2 | Baddeley & Hitch (1974) Psychol Rev 81:236 |
| cognition/decision_making.py | System 1/2, loss aversion | Kahneman & Tversky (1979) Econometrica 47:263 |
| cognition/reinforcement.py | RPE TD learning | Schultz 1997; Sutton & Barto (1998) |
| emotions/emotion_engine.py | Six basic emotions, V-A map | Ekman (1992) Cogn Emot 6:169 |
| emotions/somatic.py + W18 | Somatic markers | Damasio (1996) Phil Trans R Soc B 351:1413 |
| emotions/motivation.py | Needs stack, intrinsic drive | Maslow (1943) simplified |
| emotions/personality.py | OCEAN injection | McCrae & Costa Big Five (supporting) |
| behavior/habits.py (ADR-019) | Habit vs goal dissociation | Balleine & Dickinson (1998) Neuropharm 37:407; Yin & Knowlton (2006) Nat Rev Neurosci 7:464 |
| behavior/motor_control.py | Cerebellar error correction | Wolpert et al. forward models (supporting) |
| perception/auditory.py + integration | Multisensory enhancement | Stein & Meredith (1993) The Merging of the Senses |
| shaping (accept_maze, demo) | Potential-based shaping | Ng, Harada & Russell (1999) ICML |

## Honesty note (verbatim in paper)
Claims are COMPUTATIONAL, not phenomenal: emotions are scalar dynamics
with measurable behavioral effects; no qualia claim. Rat-scale
abstractions (grid world, tone events) are stated as such.
