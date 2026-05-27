"""Phase-3 sweep: TIC-style encoder + GyroPhase Head ablation.

Each row of the sweep is identified by ``(backbone, head_preset, seed,
channels)``. The backbone + head pair is materialised through
``PhaseAwareClassifier`` so the same training loop trains every variant.

Outputs go under ``outputs_user/imu_transition/phase3_<suffix>/seed<seed>/<row_name>``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.imu_transition.datasets.labels import CHANNEL_MODES
from experiments.imu_transition.datasets.uci_har_pt import create_dataset_splits
from experiments.imu_transition.metrics import (
    benchmark_inference,
    compute_classification_metrics,
    compute_direction_metrics,
    count_parameters,
)
from experiments.imu_transition.models.gyrophase import head_config_for
from experiments.imu_transition.models.phase_classifier import PhaseAwareClassifier
from experiments.imu_transition.utils import ensure_repo_on_path, load_config, resolve_repo_path, set_seed


# ---------------------------------------------------------------------------
# Spec table
# ---------------------------------------------------------------------------

# Each spec: (backbone_name, head_preset, pool_mode)
DEFAULT_SPECS: list[tuple[str, str, str]] = [
    # --- Baselines (Exp 1) ---
    ("tcn", "avgpool", "mean"),
    ("transformer", "avgpool", "mean"),
    ("mamba3", "avgpool", "mean"),
    # --- GyroPhase Head on backbones without internal complex state ---
    # phase feature falls back to gyro-derived per-step rotation magnitude.
    ("mamba3", "gyrophase_rd", "mean"),
    ("transformer", "gyrophase_rd", "mean"),
    # --- 2x2 SSM ablation: separate complex from selective ---
    ("real_static", "avgpool", "mean"),
    ("real_selective", "avgpool", "mean"),
    ("complex_static", "avgpool", "mean"),
    ("complex_selective", "avgpool", "mean"),
    # --- Real-Selective with GyroPhase (no internal phase, fair vs complex) ---
    ("real_selective", "gyrophase_rd", "mean"),
    # --- Complex-Selective with phase-aware heads (paper centrepiece) ---
    ("complex_selective", "phase", "mean"),
    ("complex_selective", "gyrophase", "mean"),
    ("complex_selective", "gyrophase_rd", "mean"),
    ("complex_selective", "selective_gyrophase", "mean"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase-3 GyroPhase Head sweep.")
    parser.add_argument("--config", default="experiments/imu_transition/configs/phase1.yaml")
    parser.add_argument("--seeds", nargs="*", type=int, default=[13, 42, 73])
    parser.add_argument("--channels", default="acc_gyro", choices=["acc", "gyro", "acc_gyro"])
    parser.add_argument("--task", default="direction", choices=["direction", "binary"])
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--early-stop-metric", default=None)
    parser.add_argument("--split-mode", default="random", choices=["random", "subject"])
    parser.add_argument("--specs", nargs="*", default=None,
                        help="Subset of '<backbone>.<head>[.<pool>]' specs to run.")
    parser.add_argument("--output-suffix", default="phase3")
    parser.add_argument("--device", default=None)
    parser.add_argument("--smoke", action="store_true",
                        help="Run a single seed × first spec with reduced epochs.")
    parser.add_argument("--limit-train-batches", type=int, default=None,
                        help="Cap training batches per epoch (smoke / debug).")
    return parser.parse_args()


def parse_specs(raw: list[str] | None) -> list[tuple[str, str, str]]:
    if not raw:
        return DEFAULT_SPECS
    specs: list[tuple[str, str, str]] = []
    for entry in raw:
        parts = entry.split(".")
        if len(parts) == 2:
            specs.append((parts[0], parts[1], "mean"))
        elif len(parts) == 3:
            specs.append((parts[0], parts[1], parts[2]))
        else:
            raise ValueError(f"Invalid spec '{entry}'. Use 'backbone.head[.pool]'.")
    return specs


def make_loaders(config: dict, channel_mode: str, device: torch.device):
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
        force_rebuild=False,
        split_mode=config.get("split_mode", "random"),
        task=config.get("task", "binary"),
    )
    kwargs = dict(batch_size=config["batch_size"], num_workers=config.get("num_workers", 0),
                  pin_memory=device.type == "cuda")
    if kwargs["num_workers"] > 0:
        kwargs["persistent_workers"] = True
    loaders = {
        "train": DataLoader(splits.train_dataset, shuffle=True, **kwargs),
        "val": DataLoader(splits.val_dataset, shuffle=False, **kwargs),
        "test": DataLoader(splits.test_dataset, shuffle=False, **kwargs),
    }
    return loaders, splits


def gyro_indices_for(channel_mode: str) -> tuple[int, ...] | None:
    if channel_mode == "acc_gyro":
        return (3, 4, 5)
    if channel_mode == "gyro":
        return (0, 1, 2)
    return None


def _backbone_config_for(backbone_name: str, full_config: dict) -> dict | None:
    """Map our backbone names back onto the YAML's models_config."""
    src = full_config.get("models_config", {})
    if backbone_name == "transformer":
        return src.get("transformer")
    if backbone_name == "tcn":
        return src.get("tcn")
    if backbone_name == "mamba3":
        return src.get("mamba3")
    # 2x2 ablation + real_ssm/complex_ssm share the same scaffolding
    return {"d_model": 64, "d_state": 64, "n_layers": 2, "dropout": 0.1}


def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module,
             device: torch.device, task: str, num_classes: int):
    model.eval()
    total_loss = 0.0
    all_true, all_pred = [], []
    with torch.no_grad():
        for features, labels in loader:
            features = features.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits = model(features)
            loss = criterion(logits, labels)
            total_loss += loss.item() * labels.size(0)
            preds = torch.argmax(logits, dim=-1)
            all_true.append(labels.cpu().numpy())
            all_pred.append(preds.cpu().numpy())
    y_true = np.concatenate(all_true)
    y_pred = np.concatenate(all_pred)
    if task == "direction":
        metrics = compute_direction_metrics(y_true, y_pred, num_classes=num_classes)
    else:
        metrics = compute_classification_metrics(y_true, y_pred)
    metrics["loss"] = total_loss / max(1, len(loader.dataset))
    return metrics, y_true, y_pred


def train_one(spec: tuple[str, str, str], config: dict, channel_mode: str,
              device: torch.device, run_dir: Path,
              limit_train_batches: int | None = None) -> dict:
    backbone_name, head_preset, pool_mode = spec
    set_seed(config["seed"])
    loaders, splits = make_loaders(config, channel_mode=channel_mode, device=device)

    head_cfg = head_config_for(head_preset)
    backbone_cfg = _backbone_config_for(backbone_name, config)
    model = PhaseAwareClassifier(
        backbone_name=backbone_name,
        head_config=head_cfg,
        in_channels=splits.num_channels,
        num_classes=splits.num_classes,
        gyro_indices=gyro_indices_for(channel_mode),
        backbone_config=backbone_cfg,
        dropout=0.1,
        pool=pool_mode,
    ).to(device)

    # GyroPhaseHead.classifier is lazily built on first forward — trigger one
    # forward so the optimiser sees those parameters.
    with torch.no_grad():
        sample, _ = next(iter(loaders["train"]))
        sample = sample.to(device)
        _ = model(sample)

    criterion = nn.CrossEntropyLoss(weight=splits.class_weights.to(device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"],
                                  weight_decay=config["weight_decay"])

    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = run_dir / "best.pt"
    history_path = run_dir / "history.json"
    best_metric_name = config["early_stop_metric"]
    best_metric = float("-inf")
    patience = 0
    history: list[dict] = []
    start = time.time()
    for epoch in range(1, config["epochs"] + 1):
        model.train()
        total_loss, total_n = 0.0, 0
        for batch_idx, (features, labels) in enumerate(loaders["train"]):
            if limit_train_batches is not None and batch_idx >= limit_train_batches:
                break
            features = features.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(features)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * labels.size(0)
            total_n += labels.size(0)
        train_loss = total_loss / max(1, total_n)
        val_metrics, _, _ = evaluate(model, loaders["val"], criterion, device,
                                     task=config["task"], num_classes=splits.num_classes)
        history.append({"epoch": epoch, "train_loss": train_loss, **val_metrics})
        m = float(val_metrics[best_metric_name])
        if m > best_metric:
            best_metric = m
            patience = 0
            torch.save({
                "state_dict": model.state_dict(),
                "spec": spec,
                "best_val_metrics": val_metrics,
                "epoch": epoch,
            }, ckpt_path)
        else:
            patience += 1
            if patience >= config["early_stop_patience"]:
                break
    history_path.write_text(json.dumps(history, indent=2))

    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["state_dict"])
    test_metrics, y_true, y_pred = evaluate(model, loaders["test"], criterion, device,
                                            task=config["task"], num_classes=splits.num_classes)
    sample_batch = next(iter(loaders["test"]))[0].to(device)
    inference_ms = benchmark_inference(model, sample_batch,
                                       warmup_runs=config.get("timing_warmup_runs", 25),
                                       num_runs=config.get("timing_num_runs", 100))

    elapsed = time.time() - start
    base = {k: v for k, v in test_metrics.items() if k not in ("per_class_f1", "confusion_matrix")}
    extras = {k: test_metrics[k] for k in ("per_class_f1", "confusion_matrix") if k in test_metrics}
    result = {
        "backbone": backbone_name,
        "head": head_preset,
        "pool": pool_mode,
        "channels": channel_mode,
        "seed": int(config["seed"]),
        "split_mode": config["split_mode"],
        "task": config["task"],
        "params": count_parameters(model),
        "inference_ms_per_window": inference_ms,
        "train_elapsed_sec": elapsed,
        "epochs_trained": len(history),
        **base,
    }
    (run_dir / "test_metrics.json").write_text(json.dumps({**result, **extras}, indent=2))
    (run_dir / "test_predictions.json").write_text(json.dumps({
        "y_true": y_true.astype(int).tolist(),
        "y_pred": y_pred.astype(int).tolist(),
        "test_exp_ids": splits.metadata["test_exp_ids"].astype(int).tolist(),
        "test_user_ids": splits.metadata["test_user_ids"].astype(int).tolist(),
        "test_window_starts": splits.metadata["test_window_starts"].astype(int).tolist(),
        "test_activity_ids": splits.metadata["test_activity_ids"].astype(int).tolist(),
    }))
    return result


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    config["task"] = args.task
    config["split_mode"] = args.split_mode
    config["channel_modes"] = [args.channels]
    if args.epochs is not None:
        config["epochs"] = args.epochs
    if args.early_stop_metric:
        config["early_stop_metric"] = args.early_stop_metric
    elif args.task == "direction":
        config["early_stop_metric"] = "direction_macro_f1"
    else:
        config["early_stop_metric"] = "transition_f1"
    if args.device is not None:
        config["device"] = args.device

    if args.smoke:
        args.seeds = [args.seeds[0]]
        config["epochs"] = 3
        args.specs = args.specs or ["mamba3.avgpool"]

    specs = parse_specs(args.specs)
    if args.channels not in CHANNEL_MODES:
        raise ValueError(f"Unknown channels '{args.channels}'.")

    device = torch.device(config["device"] if torch.cuda.is_available() or config["device"] == "cpu"
                          else "cpu")

    repo_root = ensure_repo_on_path()
    output_root = resolve_repo_path(config["output_root"], repo_root=repo_root) / args.output_suffix
    output_root.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    for seed in args.seeds:
        for spec in specs:
            cfg = deepcopy(config)
            cfg["seed"] = int(seed)
            run_name = f"{spec[0]}__{spec[1]}__{spec[2]}"
            seed_root = output_root / f"seed{seed}"
            seed_root.mkdir(parents=True, exist_ok=True)
            run_dir = seed_root / run_name
            print(f"\n=== seed={seed}  backbone={spec[0]}  head={spec[1]}  pool={spec[2]} ===",
                  flush=True)
            try:
                result = train_one(spec, cfg, channel_mode=args.channels, device=device,
                                   run_dir=run_dir,
                                   limit_train_batches=args.limit_train_batches)
                rows.append(result)
                short = {k: v for k, v in result.items()
                         if k in ("backbone", "head", "direction_macro_f1", "macro_f1",
                                  "transition_f1", "accuracy", "params")}
                print(json.dumps(short, indent=2), flush=True)
            except Exception as exc:                                  # noqa: BLE001
                print(f"FAILED: {exc!r}", flush=True)
                rows.append({"backbone": spec[0], "head": spec[1], "pool": spec[2],
                             "seed": int(seed), "error": str(exc)})

    csv_path = output_root / "results_phase3.csv"
    json_path = output_root / "results_phase3.json"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    json_path.write_text(json.dumps(rows, indent=2))
    print(f"\nWrote {csv_path} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
