"""exp_plan4 §4 — harder synthetic rotation tasks.

Three new tasks, all with input ``[cos θ_t, sin θ_t]`` (2 channels) so the
model has to *infer* angular velocity from phase progression rather than
reading it off an extra channel.

  direction_hard   : binary clockwise (0) vs counter-clockwise (1).
  mid_switch       : window starts at ω_a then switches to ω_b at random τ;
                     label is sign(ω_b) (= ending direction). Selective
                     scanning should help because the early half is noise.
  speed_direction6 : 6-class {slow,medium,fast} × {+,-}. Used for worst-class
                     analysis.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset


HARD_TASKS = ("direction_hard", "mid_switch", "speed_direction6")


def _sample_direction_hard(seq_len: int, noise_std: float, rng: np.random.Generator):
    theta0 = rng.uniform(-np.pi, np.pi)
    omega = rng.uniform(0.01, 0.10) * rng.choice([-1.0, 1.0])
    t = np.arange(seq_len, dtype=np.float64)
    theta = theta0 + omega * t
    x = np.stack([np.cos(theta), np.sin(theta)], axis=-1).astype(np.float32)
    label = int(omega > 0)
    if noise_std > 0:
        x += rng.normal(0.0, noise_std, size=x.shape).astype(np.float32)
    return x, label


def _sample_mid_switch(seq_len: int, noise_std: float, rng: np.random.Generator):
    theta0 = rng.uniform(-np.pi, np.pi)
    omega_a = rng.uniform(0.02, 0.10) * rng.choice([-1.0, 1.0])
    omega_b = rng.uniform(0.02, 0.10) * rng.choice([-1.0, 1.0])
    tau = int(rng.integers(seq_len // 3, seq_len * 2 // 3))
    t = np.arange(seq_len, dtype=np.float64)
    theta = np.empty_like(t)
    theta[:tau] = theta0 + omega_a * t[:tau]
    theta[tau:] = theta[tau - 1] + omega_b * (t[tau:] - (tau - 1))
    x = np.stack([np.cos(theta), np.sin(theta)], axis=-1).astype(np.float32)
    label = int(omega_b > 0)
    if noise_std > 0:
        x += rng.normal(0.0, noise_std, size=x.shape).astype(np.float32)
    return x, label


# 6-class buckets: |omega| ∈ slow ([0.01,0.04)) / medium ([0.04,0.08)) / fast ([0.08,0.12))
_SPEED_RANGES = [(0.01, 0.04), (0.04, 0.08), (0.08, 0.12)]


def _sample_speed_direction6(seq_len: int, noise_std: float, rng: np.random.Generator):
    speed_idx = int(rng.integers(0, 3))
    sign = float(rng.choice([-1.0, 1.0]))
    lo, hi = _SPEED_RANGES[speed_idx]
    omega = sign * rng.uniform(lo, hi)
    theta0 = rng.uniform(-np.pi, np.pi)
    t = np.arange(seq_len, dtype=np.float64)
    theta = theta0 + omega * t
    x = np.stack([np.cos(theta), np.sin(theta)], axis=-1).astype(np.float32)
    # label encoding: 2 * speed_idx + (sign>0)  → 0..5
    label = 2 * speed_idx + int(sign > 0)
    if noise_std > 0:
        x += rng.normal(0.0, noise_std, size=x.shape).astype(np.float32)
    return x, label


_SAMPLERS = {
    "direction_hard": (_sample_direction_hard, 2),
    "mid_switch": (_sample_mid_switch, 2),
    "speed_direction6": (_sample_speed_direction6, 6),
}


def num_classes_for(task: str) -> int:
    if task not in _SAMPLERS:
        raise ValueError(f"Unknown hard task '{task}'.")
    return _SAMPLERS[task][1]


class HardRotationDataset(Dataset):
    def __init__(self, n: int, seq_len: int, noise_std: float, task: str, seed: int):
        if task not in _SAMPLERS:
            raise ValueError(f"task must be one of {list(_SAMPLERS)}.")
        sampler, _ = _SAMPLERS[task]
        rng = np.random.default_rng(seed)
        feats = np.empty((n, seq_len, 2), dtype=np.float32)
        labels = np.empty((n,), dtype=np.int64)
        for i in range(n):
            x, y = sampler(seq_len, noise_std, rng)
            feats[i] = x
            labels[i] = y
        self.features = feats
        self.labels = labels

    def __len__(self) -> int:
        return int(self.labels.shape[0])

    def __getitem__(self, index):
        return torch.from_numpy(self.features[index]), torch.tensor(self.labels[index], dtype=torch.long)
