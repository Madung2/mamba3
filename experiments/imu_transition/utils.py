"""Common helpers for IMU transition experiment scripts."""

from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np
import torch
import yaml


def ensure_repo_on_path() -> Path:
    repo_root = Path(__file__).resolve().parents[2]
    repo_root_str = str(repo_root)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)
    return repo_root


def load_config(config_path: str | Path) -> dict:
    with Path(config_path).expanduser().open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def resolve_repo_path(path_like: str | Path, repo_root: Path | None = None) -> Path:
    path = Path(path_like).expanduser()
    if path.is_absolute():
        return path
    repo_root = repo_root or ensure_repo_on_path()
    return (repo_root / path).resolve()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_run_name(model_name: str, channel_mode: str) -> str:
    return f"{model_name}_{channel_mode}"
