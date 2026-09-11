# emotion_monitor - live dashboard of the brain's feelings.
# Simulates a scenario (struggle or reward) and draws emotion
# levels in real time. This is the 'window' into the affect system.
import sys
sys.path.insert(0, '.')
from emotions.emotion_engine import EmotionEngine

print('scenario? [struggle / reward]:')
mode = input('> ').strip().lower()
e = EmotionEngine()
print()
print(f"{'tick':>5} {'joy':>6} {'sad':>6} {'fear':>6} {'anger':>6} {'val':>7} {'aro':>6}")
print('-' * 50)
for t in range(120):
    if mode.startswith('s'):
        s = e.step(rpe=-0.6, distance_reduced=False)
    else:
        s = e.step(rpe=+0.5)
    if t % 5 == 0:
        bar = chr(9608) * int(s['sadness'] * 20)
        print(f"{t:5d} {s['joy']:6.2f} {s['sadness']:6.2f} "
              f"{s['fear']:6.2f} {s['anger']:6.2f} "
              f"{s['valence']:+7.2f} {s['arousal']:6.2f}  {bar}")
print()
print('narration:', __import__('cognition.language',
      fromlist=['RuleBasedLanguage']).RuleBasedLanguage().generate(
      {'position': (3, 4), 'attended': 'visual_cortex',
       'cortex_rate': 42.0, 'auditory_rate': 0.0,
       'emotions': {'joy': s['joy'], 'sadness': s['sadness'],
                    'fear': s['fear'], 'anger': s['anger']}}))
