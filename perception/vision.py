# Vision - MobileNet features -> visual cortex input (ADR-017).
# Pipeline: frame HxWx3 RGB -> MobileNetV3-Small 576-dim features
# (pre-classifier), L2-normalized, sparse convergent k=8 projection.
# f-I checklist arithmetic (computed BEFORE projection code):
#   dt=1ms, tau=20ms -> v moves 5% per tick toward steady state.
#   15-tick window reaches 1 - 0.95**15 = 53.7% of steady state.
#   Cross the 10mV rest-threshold gap: steady offset >= 18.6mV
#   -> minimum drive >= 0.93/tick. GAIN=3.0 verified by test.
# Weights: pretrained attempted; on failure random-init REPORTED
# in weights_source. Cadence: extract once per 50ms cycle.
from __future__ import annotations

import numpy as np

try:
    import torch
    import torchvision
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False


class VisionSystem:
    def __init__(self, seed: int = 0, n_vis_neurons: int = 100) -> None:
        self.weights_source = "unavailable"
        self.n_neurons = n_vis_neurons
        if not _HAS_TORCH:
            return
        try:
            from torchvision.models import (
                mobilenet_v3_small, MobileNet_V3_Small_Weights)
            try:
                net = mobilenet_v3_small(
                    weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1)
                self.weights_source = "pretrained"
            except Exception as e:
                net = mobilenet_v3_small(weights=None)
                self.weights_source = f"random-init ({type(e).__name__})"
            net.classifier = torch.nn.Identity()
            self.net = net.eval()
            self._torch_ok = True
        except Exception as e:
            self.weights_source = f"unavailable ({e})"
            self._torch_ok = False
        # sparse convergent projection: each vis neuron reads k=8
        rng = np.random.default_rng(seed)
        k = 8
        rows = np.repeat(np.arange(n_vis_neurons), k)
        cols = rng.integers(0, 576, size=n_vis_neurons * k)
        from scipy import sparse
        self.proj = sparse.csr_matrix(
            (np.full(n_vis_neurons * k, 2.0), (rows, cols)),
            shape=(n_vis_neurons, 576))

    def extract(self, frame: np.ndarray):
        # frame: HxWx3 uint8 RGB -> L2-normalized 576-dim float64.
        # Returns None if torch is unavailable.
        if not self.__dict__.get("_torch_ok"):
            return None
        import torch
        x = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
        x = torch.nn.functional.interpolate(
            x[None], size=(224, 224), mode="bilinear",
            align_corners=False)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        x = (x - mean) / std
        with torch.no_grad():
            f = self.net(x)[0].numpy().astype(np.float64)
        n = np.linalg.norm(f)
        return f / n if n > 0 else f

    def project(self, features):
        # features -> per-neuron drive (LifPopulation current units).
        # None features -> zero drive.
        if features is None:
            return np.zeros(self.n_neurons)
        drive = self.proj @ features
        return drive * 3.0
