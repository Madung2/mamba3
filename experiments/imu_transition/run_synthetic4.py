"""exp_plan4 §4 — harder synthetic rotation sweep.

Trains all 2x2 SSM ablation backbones plus CNN/TCN/Transformer/Mamba-3 on
the new hard synthetic tasks (no omega in input).
"""

from __future__ import annotations

import argparse
import json
import time
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

from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from experiments.imu_transition.datasets.synthetic_rotation_hard import (
    HARD_TASKS,
    HardRotationDataset,
    num_classes_for,
)
from experiments.imu_transition.metrics import benchmark_inference, count_parameters
from experiments.imu_transition.models.gyrophase import head_config_for
from experiments.imu_transition.models.phase_classifier import PhaseAwareClassifier
from experiments.imu_transition.utils import set_seed


DEFAULT_BACKBONES = [
    "cnn", "tcn", "transformer", "mamba3",
    "real_static", "real_selective",
    "complex_static", "complex_selective",
]


def evaluate(model, loader, criterion, device, num_classes: int):
    model.eval()
    total_loss, total_n = 0.0, 0
    ys, ps = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            logits = model(x)
            loss = criterion(logits, y)
            total_loss += loss.item() * y.size(0)
            total_n += y.size(0)
            ys.append(y.cpu().numpy())
            ps.append(torch.argmax(logits, dim=-1).cpu().numpy())
    y_true = np.concatenate(ys)
    y_pred = np.concatenate(ps)
    labels = list(range(num_classes))
    per_class_f1 = f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    worst_f1 = float(np.min(per_class_f1)) if num_classes > 1 else float("nan")
    cm = confusion_matrix(y_true, y_pred, labels=labels).astype(int).tolist()
    return {
        "loss": total_loss / max(1, total_n),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": macro_f1,
        "worst_class_f1": worst_f1,
        "per_class_f1": [float(v) for v in per_class_f1],
        "confusion_matrix": cm,
    }


def _backbone_config_for(backbone: str) -> dict | None:
    if backbone in {"cnn", "gru"}:
        return None
    if backbone == "tcn":
        return None
    if backbone == "transformer":
        return None
    if backbone == "mamba3":
        return None
    return {"d_model": 64, "d_state": 64, "n_layers": 2, "dropout": 0.1}


def run_one(backbone: str, task: str, seed: int, args, device: torch.device) -> dict:
    if backbone == "cnn":
        # cnn / gru are not wrapped in PhaseAwareClassifier; fall back to
        # original factory.
        from experiments.imu_transition.models.factory import create_model
        set_seed(seed)
        in_channels = 2
        num_classes = num_classes_for(task)
        model = create_model("cnn", in_channels=in_channels, num_classes=num_classes).to(device)
    else:
        set_seed(seed)
        in_channels = 2
        num_classes = num_classes_for(task)
        model = PhaseAwareClassifier(
            backbone_name=backbone,
            head_config=head_config_for("avgpool"),
            in_channels=in_channels,
            num_classes=num_classes,
            gyro_indices=None,
            backbone_config=_backbone_config_for(backbone),
            pool="mean",
        ).to(device)

    train_ds = HardRotationDataset(args.train_n, args.seq_len, args.noise_std, task, seed=seed)
    val_ds = HardRotationDataset(args.val_n, args.seq_len, args.noise_std, task, seed=seed + 1_000_000)
    test_ds = HardRotationDataset(args.test_n, args.seq_len, args.noise_std, task, seed=seed + 2_000_000)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=device.type == "cuda")
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=device.type == "cuda")
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=device.type == "cuda")

    # trigger lazy head build for PhaseAwareClassifier
    if not isinstance(model, nn.Module) or hasattr(model, "head"):
        with torch.no_grad():
            sample, _ = next(iter(train_loader))
            sample = sample.to(device)
            _ = model(sample)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()

    best_f1 = float("-inf")
    best_state = None
    patience = 0
    t0 = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
        val = evaluate(model, val_loader, criterion, device, num_classes_for(task))
        if val["macro_f1"] > best_f1:
            best_f1 = val["macro_f1"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= args.early_stop_patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    test = evaluate(model, test_loader, criterion, device, num_classes_for(task))

    first_batch = next(iter(test_loader))[0].to(device)
    inf_ms = benchmark_inference(model, first_batch, warmup_runs=10, num_runs=40)

    return {
        "backbone": backbone,
        "task": task,
        "seed": int(seed),
        "params": count_parameters(model),
        "inference_ms_per_window": inf_ms,
        "train_time_s": time.perf_counter() - t0,
        "best_val_macro_f1": best_f1,
        **{k: v for k, v in test.items() if k not in ("per_class_f1", "confusion_matrix")},
        "per_class_f1": test["per_class_f1"],
        "confusion_matrix": test["confusion_matrix"],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbones", nargs="*", default=DEFAULT_BACKBONES)
    parser.add_argument("--tasks", nargs="*", default=list(HARD_TASKS))
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
    parser.add_argument("--output-dir", default="outputs_user/imu_transition/synthetic4")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for task in args.tasks:
        for backbone in args.backbones:
            for seed in args.seeds:
                if backbone == "mamba3" and device.type != "cuda":
                    continue
                print(f"[synth4] task={task} backbone={backbone} seed={seed}", flush=True)
                try:
                    row = run_one(backbone, task, seed, args, device)
                    rows.append(row)
                    short = {k: v for k, v in row.items()
                             if k in ("backbone", "task", "seed", "macro_f1", "worst_class_f1", "accuracy")}
                    print("  -> " + json.dumps(short), flush=True)
                except Exception as exc:                                # noqa: BLE001
                    print(f"  FAILED: {exc!r}", flush=True)
                    rows.append({"backbone": backbone, "task": task, "seed": seed, "error": str(exc)})

    pd.DataFrame(rows).to_csv(out_dir / "results_synthetic4.csv", index=False)
    (out_dir / "results_synthetic4.json").write_text(json.dumps(rows, indent=2, default=float))
    print(f"\nWrote {out_dir/'results_synthetic4.csv'} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
