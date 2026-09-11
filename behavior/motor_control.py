# Motor control - cerebellar forward model (Week 23).
# The cerebellum PREDICTS the sensory consequence of each action and
# corrects error. Simple honest implementation: learned per-action
# offset between predicted and actual displacement; error drives a
# corrective nudge on the NEXT step (one-step-lag correction).
from __future__ import annotations

import numpy as np


class Cerebellum:
    def __init__(self, n_actions: int = 4, lr: float = 0.3,
                 max_correction: float = 0.4) -> None:
        self.lr = lr
        self.max_correction = max_correction
        self.predicted: dict[int, np.ndarray] = {}

    def predict(self, action: int, move: np.ndarray) -> np.ndarray:
        return self.predicted.get(action, move)

    def learn(self, action: int, predicted: np.ndarray,
              actual: np.ndarray) -> np.ndarray:
        # error between what we expected and what happened
        err = actual - predicted
        prev = self.predicted.get(action, actual * 0.0)
        self.predicted[action] = prev + self.lr * err
        return np.clip(err, -self.max_correction,
                       self.max_correction)


class MotorController:
    def __init__(self, moves: dict, cerebellum: Cerebellum | None = None,
                 noise_std: float = 0.0, seed: int = 0) -> None:
        self.moves = moves
        self.cb = cerebellum if cerebellum is not None else Cerebellum()
        self.noise_std = noise_std      # perturbed dynamics (test)
        self.rng = np.random.default_rng(seed)
        self._pending_action = None
        self._pending_pred = None

    def prepare(self, action: int) -> None:
        self._pending_action = action
        base = np.array(self.moves[action], dtype=float)
        noisy = base + self.rng.normal(0.0, self.noise_std, 2)
        self._pending_pred = base
        self._noisy = noisy

    def execute(self) -> tuple[np.ndarray, np.ndarray]:
        # world applies the NOISY move; cerebellum learns from the
        # gap and hands back a correction for the next prepare()
        pred = self.cb.predict(self._pending_action, self._pending_pred)
        err = self.cb.learn(self._pending_action, pred, self._noisy)
        return self._noisy, err
