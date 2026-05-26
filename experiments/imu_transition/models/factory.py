"""Model factory for IMU transition experiments."""

from __future__ import annotations

from copy import deepcopy

from torch import nn

from .baselines import CNN1DClassifier, GRUClassifier, TCNClassifier, TransformerEncoderClassifier
from .mamba3_classifier import Mamba3TransitionClassifier


DEFAULT_MODEL_CONFIGS = {
    "cnn": {
        "hidden_channels": [64, 128, 128],
        "kernel_sizes": [5, 3, 3],
        "dropout": 0.1,
    },
    "gru": {
        "hidden_size": 64,
        "num_layers": 2,
        "dropout": 0.1,
    },
    "tcn": {
        "channels": [64, 64, 128, 128],
        "kernel_size": 3,
        "dropout": 0.1,
    },
    "transformer": {
        "d_model": 64,
        "num_heads": 4,
        "num_layers": 2,
        "dim_feedforward": 128,
        "dropout": 0.1,
        "max_len": 256,
    },
    "mamba3": {
        "d_model": 128,
        "d_state": 64,
        "expand": 2,
        "headdim": 64,
        "n_layers": 2,
        "dropout": 0.1,
        "chunk_size": 16,
    },
}


def resolve_model_config(model_name: str, config: dict | None = None) -> dict:
    model_config = deepcopy(DEFAULT_MODEL_CONFIGS[model_name])
    if config:
        model_config.update(config)
    return model_config


def create_model(
    model_name: str,
    in_channels: int,
    num_classes: int,
    model_config: dict | None = None,
) -> nn.Module:
    cfg = resolve_model_config(model_name, config=model_config)
    if model_name == "cnn":
        return CNN1DClassifier(in_channels=in_channels, num_classes=num_classes, **cfg)
    if model_name == "gru":
        return GRUClassifier(in_channels=in_channels, num_classes=num_classes, **cfg)
    if model_name == "tcn":
        return TCNClassifier(in_channels=in_channels, num_classes=num_classes, **cfg)
    if model_name == "transformer":
        return TransformerEncoderClassifier(in_channels=in_channels, num_classes=num_classes, **cfg)
    if model_name == "mamba3":
        return Mamba3TransitionClassifier(in_channels=in_channels, num_classes=num_classes, **cfg)
    if model_name == "mamba2":
        raise NotImplementedError("Mamba-2 ablation is reserved for the next experiment phase.")
    raise ValueError(f"Unknown model '{model_name}'.")
