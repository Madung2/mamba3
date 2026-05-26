"""Metrics and runtime utilities for IMU transition experiments."""

from __future__ import annotations

import time

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support


def compute_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Compute aggregate metrics for binary transition detection."""

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="binary",
        pos_label=1,
        zero_division=0,
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "transition_precision": float(precision),
        "transition_recall": float(recall),
        "transition_f1": float(f1),
    }


def count_parameters(model: torch.nn.Module) -> int:
    return int(sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad))


def benchmark_inference(
    model: torch.nn.Module,
    batch: torch.Tensor,
    warmup_runs: int = 25,
    num_runs: int = 100,
) -> float:
    """Measure average per-window inference time in milliseconds."""

    model.eval()
    batch = batch.detach()
    with torch.no_grad():
        if batch.is_cuda:
            torch.cuda.synchronize(batch.device)
        for _ in range(warmup_runs):
            _ = model(batch)
        if batch.is_cuda:
            torch.cuda.synchronize(batch.device)

        start = time.perf_counter()
        for _ in range(num_runs):
            _ = model(batch)
        if batch.is_cuda:
            torch.cuda.synchronize(batch.device)
        elapsed = time.perf_counter() - start

    return float((elapsed / num_runs) * 1000.0 / batch.shape[0])
