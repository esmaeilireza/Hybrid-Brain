# ADR-017: Vision Stack - CPU Torch, Quantized MobileNet, PyVista OpenGL

## Status: Accepted | Week 21

## Decisions
1. Torch CPU build (GTX 960M: 4GB VRAM; single-frame CPU inference ~10ms
   beats transfer overhead; GPU deferred until profiling demands it).
2. MobileNetV3-Small, FEATURE VECTOR ONLY (pre-classifier, 576-dim) -
   no ImageNet classes; the cortex learns meaning via the value system.
   Pretrained net = perception primitives; brain = meaning.
3. PyVista offscreen OpenGL rendering (GPU helps rendering only).
4. Frame cadence: every cognitive cycle (50ms), not every tick.
5. Projection discipline: sparse convergent k=8, f-I gain calibration
   computed in docstring BEFORE projection code (dilution-bug checklist).

## Fallbacks (pre-committed)
GL headless fail -> 480x360 -> Plotly/WebGL. Inference over budget ->
160x160 input.
