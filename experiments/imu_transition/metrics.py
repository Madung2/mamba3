"""Metrics and runtime utilities for IMU transition experiments."""

from __future__ import annotations

import time

import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support


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


def compute_direction_metrics(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int = 7) -> dict[str, float]:
    """Multiclass metrics for the 7-class direction task.

    Class 0 = non-transition; classes 1..num_classes-1 = directed transitions.
    `direction_macro_f1` averages only over the transition classes (1..6),
    which is the primary signal for direction-aware evaluation.
    """

    f1_per_class = f1_score(y_true, y_pred, labels=list(range(num_classes)), average=None, zero_division=0)
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    direction_macro_f1 = float(np.mean(f1_per_class[1:])) if num_classes > 1 else float("nan")
    worst_direction_f1 = float(np.min(f1_per_class[1:])) if num_classes > 1 else float("nan")
    non_trans_f1 = float(f1_per_class[0])

    # Treat any predicted transition class as "transition" for binary-style
    # transition_precision / transition_recall so that direction results stay
    # comparable to the phase-1 binary tables.
    y_true_bin = (y_true >= 1).astype(np.int64)
    y_pred_bin = (y_pred >= 1).astype(np.int64)
    p, r, f1, _ = precision_recall_fscore_support(
        y_true_bin, y_pred_bin, average="binary", pos_label=1, zero_division=0,
    )

    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "non_transition_f1": non_trans_f1,
        "direction_macro_f1": direction_macro_f1,
        "worst_direction_f1": worst_direction_f1,
        "transition_precision": float(p),
        "transition_recall": float(r),
        "transition_f1": float(f1),
        "per_class_f1": [float(v) for v in f1_per_class],
        "confusion_matrix": cm.astype(int).tolist(),
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
