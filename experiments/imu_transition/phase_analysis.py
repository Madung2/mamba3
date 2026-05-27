"""Hidden-phase vs gyro-magnitude analysis for Complex-SSM checkpoints.

Reload a trained ComplexSSMClassifier, forward the test split with
`expose_hidden=True`, recover per-step (real, imag) from the *last* block,
compute |Δphase|, and correlate with the input gyro magnitude.

Outputs per run:
  outputs_user/imu_transition/phase_analysis/<run_id>/
    summary.json              # aggregate stats
    per_window.csv            # one row per window: y_true, mean |dphase|, mean gyro
"""

from __future__ import annotations

import argparse
import glob
import json
import math
from copy import deepcopy
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.imu_transition.datasets.uci_har_pt import create_dataset_splits
from experiments.imu_transition.models.factory import create_model, resolve_model_config
from experiments.imu_transition.models.ssm_ablation import ComplexSSMBlock, ComplexSSMClassifier
from experiments.imu_transition.utils import resolve_repo_path


def wrap_to_pi(x: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(x), torch.cos(x))


def build_test_split(checkpoint: dict) -> object:
    cfg = checkpoint["config"]
    splits = create_dataset_splits(
        data_root=resolve_repo_path(cfg["data_root"]),
        window_size=int(cfg["window_size"]),
        stride=int(cfg["stride"]),
        channel_mode=checkpoint["channel_mode"],
        train_ratio=cfg["train_ratio"],
        val_ratio=cfg["val_ratio"],
        test_ratio=cfg["test_ratio"],
        seed=int(cfg["seed"]),
        normalize=cfg.get("normalize", True),
        subset_fraction=cfg.get("subset_fraction", 1.0),
        force_rebuild=False,
        split_mode=cfg.get("split_mode", "random"),
        task=cfg.get("task", "binary"),
    )
    return splits


def channel_to_gyro_indices(channel_mode: str) -> list[int]:
    """Within the model's already-sliced channel set, which indices are gyro?"""
    if channel_mode == "gyro":
        return [0, 1, 2]
    if channel_mode == "acc_gyro":
        return [3, 4, 5]
    return []  # acc_only — no gyro available


def analyze_checkpoint(ckpt_path: Path, device: torch.device, batch_size: int = 64) -> dict[str, object]:
    checkpoint = torch.load(ckpt_path, map_location=device)
    if checkpoint["model_name"] != "complex_ssm":
        raise SystemExit(f"Expected complex_ssm checkpoint, got {checkpoint['model_name']}")

    splits = build_test_split(checkpoint)
    channel_mode = checkpoint["channel_mode"]
    gyro_idx = channel_to_gyro_indices(channel_mode)

    cfg = resolve_model_config("complex_ssm", config=checkpoint.get("model_config"))
    model = ComplexSSMClassifier(in_channels=splits.num_channels, num_classes=splits.num_classes,
                                 expose_hidden=True, **{k: v for k, v in cfg.items()
                                                         if k in ("d_model", "d_state", "n_layers", "dropout")})
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device).eval()
    # Make sure all blocks expose hidden state regardless of cfg
    for block in model.blocks:
        if isinstance(block, ComplexSSMBlock):
            block.expose_hidden = True

    test_loader = torch.utils.data.DataLoader(splits.test_dataset, batch_size=batch_size, shuffle=False)

    per_window_rows: list[dict[str, float]] = []
    transition_window_idx = 0
    with torch.no_grad():
        for features, labels in test_loader:
            features = features.to(device)
            _ = model(features)
            last_block = model.blocks[-1]
            real = last_block._last_hidden_real  # (B, T, d_state)
            imag = last_block._last_hidden_imag  # (B, T, d_state)
            if real is None or imag is None:
                raise RuntimeError("ComplexSSMBlock did not expose hidden state.")
            phase = torch.atan2(imag, real)                       # (B, T, d_state)
            dphase = wrap_to_pi(phase[:, 1:] - phase[:, :-1])     # (B, T-1, d_state)
            mean_dphase = dphase.abs().mean(dim=(1, 2))           # (B,)

            if gyro_idx:
                gyro = features[:, :, gyro_idx]                    # (B, T, 3)
                gyro_mag = torch.sqrt((gyro ** 2).sum(dim=-1))    # (B, T)
                mean_gyro = gyro_mag.mean(dim=-1)                  # (B,)
            else:
                mean_gyro = torch.full_like(mean_dphase, float("nan"))

            mean_dphase = mean_dphase.cpu().numpy()
            mean_gyro = mean_gyro.cpu().numpy()
            labels_np = labels.cpu().numpy()
            for i in range(labels_np.shape[0]):
                per_window_rows.append({
                    "y_true": int(labels_np[i]),
                    "mean_abs_delta_phase": float(mean_dphase[i]),
                    "mean_gyro_magnitude": float(mean_gyro[i]),
                })

    df = pd.DataFrame(per_window_rows)
    # aggregate stats
    summary = {
        "checkpoint": str(ckpt_path),
        "channel_mode": channel_mode,
        "task": str(checkpoint["config"].get("task", "binary")),
        "n_test": int(len(df)),
    }
    if (df["y_true"] >= 1).any() and (df["y_true"] == 0).any():
        tr = df[df["y_true"] >= 1]["mean_abs_delta_phase"]
        ntr = df[df["y_true"] == 0]["mean_abs_delta_phase"]
        summary["delta_phase_mean_transition"] = float(tr.mean())
        summary["delta_phase_std_transition"] = float(tr.std(ddof=1)) if len(tr) > 1 else 0.0
        summary["delta_phase_median_transition"] = float(tr.median())
        summary["delta_phase_mean_non_transition"] = float(ntr.mean())
        summary["delta_phase_std_non_transition"] = float(ntr.std(ddof=1)) if len(ntr) > 1 else 0.0
        summary["delta_phase_median_non_transition"] = float(ntr.median())
        summary["delta_phase_ratio"] = float(tr.mean() / max(ntr.mean(), 1e-12))
    if gyro_idx and df["mean_gyro_magnitude"].notna().any():
        corr = float(df[["mean_abs_delta_phase", "mean_gyro_magnitude"]].corr().iloc[0, 1])
        summary["phase_gyro_pearson_r"] = corr
        for cls in sorted(df["y_true"].unique().tolist()):
            sub = df[df["y_true"] == cls]
            if len(sub) >= 5:
                summary[f"phase_gyro_pearson_r_class_{cls}"] = float(
                    sub[["mean_abs_delta_phase", "mean_gyro_magnitude"]].corr().iloc[0, 1]
                )
    return {"summary": summary, "per_window": df}


def main():
    parser = argparse.ArgumentParser(description="Hidden phase analysis (complex_ssm).")
    parser.add_argument("--checkpoint-glob", required=True, help="Glob of best.pt files for complex_ssm runs.")
    parser.add_argument("--output-dir", default="outputs_user/imu_transition/phase_analysis")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    device = torch.device(args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    checkpoints = sorted(Path(p) for p in glob.glob(args.checkpoint_glob, recursive=True))
    if not checkpoints:
        raise SystemExit(f"No checkpoints matched: {args.checkpoint_glob}")

    all_summaries = []
    for ckpt in checkpoints:
        try:
            run_id = "__".join(ckpt.parts[-4:-1])
            print(f"Analyzing {run_id}")
            result = analyze_checkpoint(ckpt, device=device, batch_size=args.batch_size)
            sub_dir = out_dir / run_id
            sub_dir.mkdir(parents=True, exist_ok=True)
            (sub_dir / "summary.json").write_text(json.dumps(result["summary"], indent=2), encoding="utf-8")
            result["per_window"].to_csv(sub_dir / "per_window.csv", index=False)
            summary = dict(result["summary"])
            summary["run_id"] = run_id
            all_summaries.append(summary)
        except Exception as exc:
            print(f"  ERROR: {exc}")

    if all_summaries:
        agg = pd.DataFrame(all_summaries)
        agg.to_csv(out_dir / "all_summaries.csv", index=False)
        print(f"\nWrote {out_dir / 'all_summaries.csv'} ({len(all_summaries)} runs)")
        print(agg.to_string(index=False))


if __name__ == "__main__":
    main()
