"""Training entry point for IMU transition detection experiments."""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path
import sys

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.imu_transition.datasets.uci_har_pt import create_dataset_splits
from experiments.imu_transition.metrics import (
    benchmark_inference,
    compute_classification_metrics,
    compute_direction_metrics,
    count_parameters,
)
from experiments.imu_transition.models.factory import create_model, resolve_model_config
from experiments.imu_transition.utils import build_run_name, ensure_repo_on_path, load_config, resolve_repo_path, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a binary IMU transition detector.")
    parser.add_argument(
        "--config",
        default="experiments/imu_transition/configs/phase1.yaml",
        help="Path to the YAML config file.",
    )
    parser.add_argument("--model", required=True, choices=["cnn", "gru", "tcn", "transformer", "mamba3"])
    parser.add_argument("--channels", required=True, choices=["acc", "gyro", "acc_gyro"])
    parser.add_argument("--device", default=None, help="Override device from config.")
    parser.add_argument("--epochs", type=int, default=None, help="Override epoch count.")
    parser.add_argument("--subset-fraction", type=float, default=None, help="Override subset fraction.")
    parser.add_argument("--force-rebuild", action="store_true", help="Rebuild cached windows.")
    return parser.parse_args()


def resolve_device(requested_device: str, model_name: str) -> torch.device:
    if requested_device.startswith("cuda") and not torch.cuda.is_available():
        if model_name == "mamba3":
            raise RuntimeError("Mamba-3 experiments require CUDA, but CUDA is not available.")
        return torch.device("cpu")
    return torch.device(requested_device)


def build_dataloaders(config: dict, channel_mode: str, device: torch.device) -> tuple[dict[str, DataLoader], object]:
    repo_root = ensure_repo_on_path()
    splits = create_dataset_splits(
        data_root=resolve_repo_path(config["data_root"], repo_root=repo_root),
        window_size=config["window_size"],
        stride=config["stride"],
        channel_mode=channel_mode,
        train_ratio=config["train_ratio"],
        val_ratio=config["val_ratio"],
        test_ratio=config["test_ratio"],
        seed=config["seed"],
        normalize=config.get("normalize", True),
        subset_fraction=config.get("subset_fraction", 1.0),
        force_rebuild=config.get("force_rebuild", False),
        split_mode=config.get("split_mode", "random"),
        task=config.get("task", "binary"),
    )
    loader_kwargs = {
        "batch_size": config["batch_size"],
        "num_workers": config.get("num_workers", 0),
        "pin_memory": device.type == "cuda",
    }
    if loader_kwargs["num_workers"] > 0:
        loader_kwargs["persistent_workers"] = True

    loaders = {
        "train": DataLoader(splits.train_dataset, shuffle=True, **loader_kwargs),
        "val": DataLoader(splits.val_dataset, shuffle=False, **loader_kwargs),
        "test": DataLoader(splits.test_dataset, shuffle=False, **loader_kwargs),
    }
    return loaders, splits


def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    task: str = "binary",
    num_classes: int = 2,
) -> tuple[dict[str, float], np.ndarray, np.ndarray]:
    model.eval()
    total_loss = 0.0
    all_labels: list[np.ndarray] = []
    all_preds: list[np.ndarray] = []

    with torch.no_grad():
        for features, labels in loader:
            features = features.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits = model(features)
            loss = criterion(logits, labels)
            total_loss += loss.item() * labels.size(0)
            preds = torch.argmax(logits, dim=-1)
            all_labels.append(labels.detach().cpu().numpy())
            all_preds.append(preds.detach().cpu().numpy())

    y_true = np.concatenate(all_labels)
    y_pred = np.concatenate(all_preds)
    if task == "direction":
        metrics = compute_direction_metrics(y_true, y_pred, num_classes=num_classes)
    else:
        metrics = compute_classification_metrics(y_true, y_pred)
    metrics["loss"] = total_loss / max(1, len(loader.dataset))
    return metrics, y_true, y_pred


def train_single_experiment(config: dict, model_name: str, channel_mode: str) -> dict[str, object]:
    config = deepcopy(config)
    set_seed(config["seed"])

    device = resolve_device(config["device"], model_name=model_name)
    loaders, splits = build_dataloaders(config, channel_mode=channel_mode, device=device)

    model_cfg = resolve_model_config(model_name, config.get("models_config", {}).get(model_name, {}))
    model = create_model(
        model_name=model_name,
        in_channels=splits.num_channels,
        num_classes=splits.num_classes,
        model_config=model_cfg,
    ).to(device)

    class_weights = splits.class_weights.to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["lr"],
        weight_decay=config["weight_decay"],
    )

    run_name = build_run_name(model_name, channel_mode)
    repo_root = ensure_repo_on_path()
    output_root = resolve_repo_path(config["output_root"], repo_root=repo_root)
    run_dir = output_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = run_dir / "best.pt"
    history_path = run_dir / "history.json"

    best_metric_name = config.get("early_stop_metric", "transition_f1")
    best_metric = float("-inf")
    patience = 0
    history: list[dict[str, float]] = []

    for epoch in range(1, config["epochs"] + 1):
        model.train()
        total_loss = 0.0
        total_examples = 0

        for features, labels in loaders["train"]:
            features = features.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            logits = model(features)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            total_examples += batch_size

        train_loss = total_loss / max(1, total_examples)
        val_metrics, _, _ = evaluate_model(
            model, loaders["val"], criterion, device,
            task=config.get("task", "binary"), num_classes=splits.num_classes,
        )
        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            **val_metrics,
        }
        history.append(record)

        metric_value = float(val_metrics[best_metric_name])
        if metric_value > best_metric:
            best_metric = metric_value
            patience = 0
            torch.save(
                {
                    "model_name": model_name,
                    "channel_mode": channel_mode,
                    "model_config": model_cfg,
                    "config": config,
                    "state_dict": model.state_dict(),
                    "normalization": {
                        key: value.tolist() for key, value in splits.normalization.items()
                    },
                    "split_sizes": splits.split_sizes,
                    "best_val_metrics": val_metrics,
                },
                checkpoint_path,
            )
        else:
            patience += 1
            if patience >= config["early_stop_patience"]:
                break

    history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    test_metrics, y_true, y_pred = evaluate_model(
        model, loaders["test"], criterion, device,
        task=config.get("task", "binary"), num_classes=splits.num_classes,
    )

    first_batch = next(iter(loaders["test"]))[0].to(device)
    inference_ms = benchmark_inference(
        model,
        first_batch,
        warmup_runs=config.get("timing_warmup_runs", 25),
        num_runs=config.get("timing_num_runs", 100),
    )

    results = {
        "model": model_name,
        "channels": channel_mode,
        "params": count_parameters(model),
        "inference_ms_per_window": inference_ms,
        "checkpoint_path": str(checkpoint_path),
        "split_sizes": splits.split_sizes,
        "best_val_metrics": checkpoint["best_val_metrics"],
        "seed": int(config["seed"]),
        "window_size": int(config["window_size"]),
        "stride": int(config["stride"]),
        "split_mode": str(config.get("split_mode", "random")),
        "task": str(config.get("task", "binary")),
        "num_classes": int(splits.num_classes),
        **{k: v for k, v in test_metrics.items() if k not in ("per_class_f1", "confusion_matrix")},
    }
    extras = {k: test_metrics[k] for k in ("per_class_f1", "confusion_matrix") if k in test_metrics}

    (run_dir / "test_metrics.json").write_text(
        json.dumps(
            {
                **results,
                **extras,
                "y_true_shape": int(y_true.shape[0]),
                "y_pred_shape": int(y_pred.shape[0]),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    (run_dir / "test_predictions.json").write_text(
        json.dumps(
            {
                "model": model_name,
                "channels": channel_mode,
                "seed": int(config["seed"]),
                "window_size": int(config["window_size"]),
                "stride": int(config["stride"]),
                "split_mode": str(config.get("split_mode", "random")),
                "y_true": y_true.astype(int).tolist(),
                "y_pred": y_pred.astype(int).tolist(),
                "test_exp_ids": splits.metadata["test_exp_ids"].astype(int).tolist(),
                "test_user_ids": splits.metadata["test_user_ids"].astype(int).tolist(),
                "test_window_starts": splits.metadata["test_window_starts"].astype(int).tolist(),
                "test_activity_ids": splits.metadata["test_activity_ids"].astype(int).tolist(),
            }
        ),
        encoding="utf-8",
    )
    return results


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.device is not None:
        config["device"] = args.device
    if args.epochs is not None:
        config["epochs"] = args.epochs
    if args.subset_fraction is not None:
        config["subset_fraction"] = args.subset_fraction
    config["force_rebuild"] = args.force_rebuild

    results = train_single_experiment(config=config, model_name=args.model, channel_mode=args.channels)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
