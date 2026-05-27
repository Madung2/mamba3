"""Synthetic rotation sequences for the complex-SSM controlled ablation.

Each sample is a (seq_len, 3) tensor of [cos θ_t, sin θ_t, ω_t] plus optional
Gaussian noise. Three tasks are supported:

  direction   : binary clockwise (0) vs counter-clockwise (1) classification
  phase_jump  : binary detection of a phase discontinuity inside the window
  speed_change: binary detection of an angular-velocity change inside the window

Used by run_synthetic.py (experiment 4 in exp_plan2.md).
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset


TASKS = ("direction", "phase_jump", "speed_change")


def generate_sample(seq_len: int, noise_std: float, task: str, rng: np.random.Generator) -> tuple[np.ndarray, int]:
    theta0 = rng.uniform(-np.pi, np.pi)
    omega = rng.uniform(0.02, 0.20) * rng.choice([-1.0, 1.0])
    t = np.arange(seq_len, dtype=np.float64)
    theta = theta0 + omega * t
    omega_track = np.full_like(theta, omega)

    if task == "direction":
        label = int(omega > 0)
    elif task == "phase_jump":
        has_jump = rng.random() < 0.5
        if has_jump:
            tau = int(rng.integers(seq_len // 4, max(seq_len // 4 + 1, seq_len * 3 // 4)))
            delta = rng.uniform(np.pi / 4, np.pi) * rng.choice([-1.0, 1.0])
            theta[t >= tau] += delta
        label = int(has_jump)
    elif task == "speed_change":
        has_change = rng.random() < 0.5
        if has_change:
            tau = int(rng.integers(seq_len // 4, max(seq_len // 4 + 1, seq_len * 3 // 4)))
            omega2 = rng.uniform(0.02, 0.20) * rng.choice([-1.0, 1.0])
            theta[t >= tau] = theta[tau - 1] + omega2 * (t[t >= tau] - (tau - 1))
            omega_track[t >= tau] = omega2
        label = int(has_change)
    else:
        raise ValueError(f"Unknown synthetic task '{task}'.")

    x = np.stack([np.cos(theta), np.sin(theta), omega_track], axis=-1).astype(np.float32)
    if noise_std > 0:
        x = x + rng.normal(0.0, noise_std, size=x.shape).astype(np.float32)
    return x, label


class SyntheticRotationDataset(Dataset):
    def __init__(self, n: int, seq_len: int, noise_std: float, task: str, seed: int):
        if task not in TASKS:
            raise ValueError(f"task must be one of {TASKS}, got {task!r}")
        rng = np.random.default_rng(seed)
        feats = np.empty((n, seq_len, 3), dtype=np.float32)
        labels = np.empty((n,), dtype=np.int64)
        for i in range(n):
            x, y = generate_sample(seq_len, noise_std, task, rng)
            feats[i] = x
            labels[i] = y
        self.features = feats
        self.labels = labels

    def __len__(self) -> int:
        return int(self.labels.shape[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.from_numpy(self.features[index]), torch.tensor(self.labels[index], dtype=torch.long)
