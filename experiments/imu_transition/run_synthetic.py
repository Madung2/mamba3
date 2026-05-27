"""Synthetic-rotation training sweep (exp_plan2.md experiment 4).

Each (model × task × seed) combo trains a binary classifier on synthetic
rotation sequences and writes test metrics to a unified CSV.
"""

from __future__ import annotations

import argparse
import json
import time
from copy import deepcopy
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.imu_transition.datasets.synthetic_rotation import SyntheticRotationDataset, TASKS
from experiments.imu_transition.metrics import (
    benchmark_inference,
    compute_classification_metrics,
    count_parameters,
)
from experiments.imu_transition.models.factory import create_model, resolve_model_config
from experiments.imu_transition.utils import set_seed


DEFAULT_MODELS = ("cnn", "tcn", "transformer", "real_ssm", "complex_ssm", "mamba3")


def evaluate(model, loader, criterion, device) -> tuple[dict[str, float], np.ndarray, np.ndarray]:
    model.eval()
    total_loss = 0.0
    ys: list[np.ndarray] = []
    ps: list[np.ndarray] = []
    with torch.no_grad():
        for features, labels in loader:
            features = features.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits = model(features)
            loss = criterion(logits, labels)
            total_loss += loss.item() * labels.size(0)
            preds = torch.argmax(logits, dim=-1)
            ys.append(labels.detach().cpu().numpy())
            ps.append(preds.detach().cpu().numpy())
    y_true = np.concatenate(ys)
    y_pred = np.concatenate(ps)
    metrics = compute_classification_metrics(y_true, y_pred)
    metrics["loss"] = total_loss / max(1, len(loader.dataset))
    return metrics, y_true, y_pred


def run_one(args, model_name: str, task: str, seed: int, device: torch.device) -> dict[str, object]:
    set_seed(seed)
    train_ds = SyntheticRotationDataset(args.train_n, args.seq_len, args.noise_std, task, seed=seed)
    val_ds = SyntheticRotationDataset(args.val_n, args.seq_len, args.noise_std, task, seed=seed + 100_000)
    test_ds = SyntheticRotationDataset(args.test_n, args.seq_len, args.noise_std, task, seed=seed + 200_000)
    loader_kwargs = {"batch_size": args.batch_size, "num_workers": args.num_workers,
                     "pin_memory": device.type == "cuda"}
    if loader_kwargs["num_workers"] > 0:
        loader_kwargs["persistent_workers"] = True
    train_loader = DataLoader(train_ds, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_ds, shuffle=False, **loader_kwargs)

    model_cfg = resolve_model_config(model_name, config=None)
    # synthetic inputs are 3-channel
    model = create_model(model_name=model_name, in_channels=3, num_classes=2, model_config=model_cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()

    best_val_f1 = float("-inf")
    best_state = None
    patience = 0
    t0 = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        for features, labels in train_loader:
            features = features.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(features)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
        val_metrics, _, _ = evaluate(model, val_loader, criterion, device)
        if val_metrics["macro_f1"] > best_val_f1:
            best_val_f1 = float(val_metrics["macro_f1"])
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= args.early_stop_patience:
                break
    train_time = time.perf_counter() - t0

    if best_state is not None:
        model.load_state_dict(best_state)
    test_metrics, _, _ = evaluate(model, test_loader, criterion, device)

    first_batch = next(iter(test_loader))[0].to(device)
    inference_ms = benchmark_inference(
        model, first_batch, warmup_runs=args.timing_warmup_runs, num_runs=args.timing_num_runs,
    )

    return {
        "model": model_name,
        "task": task,
        "seed": int(seed),
        "seq_len": int(args.seq_len),
        "noise_std": float(args.noise_std),
        "params": count_parameters(model),
        "inference_ms_per_window": inference_ms,
        "train_time_s": train_time,
        "train_size": args.train_n,
        "val_size": args.val_n,
        "test_size": args.test_n,
        "best_val_macro_f1": best_val_f1,
        **test_metrics,
    }


def main():
    parser = argparse.ArgumentParser(description="Synthetic rotation sweep (experiment 4).")
    parser.add_argument("--models", nargs="*", default=list(DEFAULT_MODELS))
    parser.add_argument("--tasks", nargs="*", default=list(TASKS))
    parser.add_argument("--seeds", nargs="*", type=int, default=[13, 42, 73])
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--noise-std", type=float, default=0.05)
    parser.add_argument("--train-n", type=int, default=10000)
    parser.add_argument("--val-n", type=int, default=2000)
    parser.add_argument("--test-n", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--early-stop-patience", type=int, default=6)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--timing-warmup-runs", type=int, default=10)
    parser.add_argument("--timing-num-runs", type=int, default=40)
    parser.add_argument("--output-dir", default="outputs_user/imu_transition/synthetic")
    args = parser.parse_args()

    device = torch.device(args.device if (not args.device.startswith("cuda")) or torch.cuda.is_available() else "cpu")

    rows: list[dict[str, object]] = []
    for task in args.tasks:
        for model_name in args.models:
            for seed in args.seeds:
                print(f"[synthetic] task={task} model={model_name} seed={seed}", flush=True)
                # Mamba-3 requires CUDA; gracefully skip if device is CPU.
                if model_name == "mamba3" and device.type != "cuda":
                    print("  ... skipping mamba3 on CPU")
                    continue
                row = run_one(args, model_name, task, seed, device)
                rows.append(row)
                print(f"  -> acc={row['accuracy']:.4f}  macro_f1={row['macro_f1']:.4f}  "
                      f"ms/win={row['inference_ms_per_window']:.4f}", flush=True)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(out_dir / "results_synthetic.csv", index=False)
    (out_dir / "results_synthetic.json").write_text(json.dumps(rows, indent=2, default=float), encoding="utf-8")
    print(f"\n=== Synthetic sweep finished: {len(rows)} rows -> {out_dir} ===")
    print(frame.to_string(index=False))


if __name__ == "__main__":
    main()
