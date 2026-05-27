"""Aggregate phase-2 sweeps (direction, ssm-ablation, synthetic, phase-analysis)
into mean±std markdown tables consumable by result.md.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def fmt(mean: float, std: float, digits: int = 4) -> str:
    if np.isnan(mean):
        return "nan"
    return f"{mean:.{digits}f} ± {std:.{digits}f}"


def agg(df: pd.DataFrame, group: list[str], metrics: list[str]) -> pd.DataFrame:
    rows = []
    for key, sub in df.groupby(group, sort=True):
        if not isinstance(key, tuple):
            key = (key,)
        row: dict[str, object] = dict(zip(group, key))
        row["n_seeds"] = int(sub["seed"].nunique())
        for m in metrics:
            arr = sub[m].astype(float).to_numpy()
            arr = arr[~np.isnan(arr)] if np.issubdtype(arr.dtype, np.floating) else arr
            row[f"{m}_mean"] = float(arr.mean()) if arr.size else float("nan")
            row[f"{m}_std"] = float(arr.std(ddof=1)) if arr.size > 1 else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def render(df: pd.DataFrame, columns: list[tuple[str, str, int]], group_cols: list[str]) -> str:
    lines = []
    header = " | ".join(group_cols + [c[0] for c in columns])
    sep = " | ".join(["---"] * (len(group_cols) + len(columns)))
    lines.append("| " + header + " |")
    lines.append("| " + sep + " |")
    for _, r in df.iterrows():
        cells = [str(r[c]) for c in group_cols]
        for label, metric, digits in columns:
            cells.append(fmt(r[f"{metric}_mean"], r[f"{metric}_std"], digits))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def aggregate_per_class_f1(direction_dir: Path) -> pd.DataFrame:
    """Look into per-run test_metrics.json for per_class_f1, average across seeds."""
    from collections import defaultdict
    bucket: dict[tuple[str, str], list[list[float]]] = defaultdict(list)
    for jf in direction_dir.glob("seed*/*/test_metrics.json"):
        d = json.loads(jf.read_text(encoding="utf-8"))
        if "per_class_f1" not in d:
            continue
        bucket[(d["model"], d["channels"])].append(d["per_class_f1"])
    rows = []
    for (model, channels), arrs in bucket.items():
        a = np.asarray(arrs, dtype=float)  # (n_seeds, 7)
        mean = a.mean(axis=0)
        std = a.std(axis=0, ddof=1) if a.shape[0] > 1 else np.zeros_like(mean)
        rows.append({"model": model, "channels": channels, "n_seeds": a.shape[0],
                     **{f"f1_class_{i}_mean": float(mean[i]) for i in range(a.shape[1])},
                     **{f"f1_class_{i}_std": float(std[i]) for i in range(a.shape[1])}})
    return pd.DataFrame(rows)


def render_per_class(df: pd.DataFrame, class_names: list[str]) -> str:
    cols = [f1 for f1 in df.columns if f1.startswith("f1_class_")]
    n_cls = sum(1 for c in cols if c.endswith("_mean"))
    header = ["model", "channels"] + class_names
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join(["---"] * len(header)) + " |"]
    for _, r in df.iterrows():
        cells = [str(r["model"]), str(r["channels"])]
        for i in range(n_cls):
            cells.append(fmt(r[f"f1_class_{i}_mean"], r[f"f1_class_{i}_std"], 3))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def average_confusion_matrix(direction_dir: Path, model: str, channels: str) -> np.ndarray | None:
    mats = []
    for jf in direction_dir.glob(f"seed*/{model}_{channels}/test_metrics.json"):
        d = json.loads(jf.read_text(encoding="utf-8"))
        if "confusion_matrix" in d:
            mats.append(np.asarray(d["confusion_matrix"], dtype=float))
    if not mats:
        return None
    return np.mean(np.stack(mats, axis=0), axis=0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="outputs_user/imu_transition")
    p.add_argument("--out", default="outputs_user/imu_transition/agg_phase2.md")
    args = p.parse_args()

    root = Path(args.root)
    out_chunks: dict[str, str] = {}

    class_names = ["non_trans", "stand_to_sit", "sit_to_stand", "sit_to_lie",
                   "lie_to_sit", "stand_to_lie", "lie_to_stand"]

    # ===== Direction sweep =====
    dir_csv = root / "direction_5seed" / "results_phase1.csv"
    if dir_csv.exists():
        df = pd.read_csv(dir_csv)
        t = agg(df, ["model", "channels"], [
            "accuracy", "macro_f1", "weighted_f1", "non_transition_f1",
            "direction_macro_f1", "worst_direction_f1",
            "transition_precision", "transition_recall", "transition_f1",
            "inference_ms_per_window",
        ])
        out_chunks["direction_main"] = render(t, [
            ("Acc", "accuracy", 4),
            ("Macro F1", "macro_f1", 4),
            ("Non-trans F1", "non_transition_f1", 4),
            ("**Direction Macro F1**", "direction_macro_f1", 4),
            ("Worst-class F1", "worst_direction_f1", 4),
            ("Trans F1 (binarized)", "transition_f1", 4),
            ("ms/win", "inference_ms_per_window", 4),
        ], ["model", "channels"])
        pcf = aggregate_per_class_f1(root / "direction_5seed")
        if not pcf.empty:
            out_chunks["direction_per_class"] = render_per_class(pcf, class_names)

    # ===== Real vs Complex SSM =====
    abl_csv = root / "ssm_ablation_5seed" / "results_phase1.csv"
    if abl_csv.exists():
        df = pd.read_csv(abl_csv)
        t = agg(df, ["model", "channels"], [
            "accuracy", "macro_f1", "direction_macro_f1", "worst_direction_f1",
            "transition_f1", "params", "inference_ms_per_window",
        ])
        out_chunks["ssm_ablation"] = render(t, [
            ("Acc", "accuracy", 4),
            ("Macro F1", "macro_f1", 4),
            ("**Direction Macro F1**", "direction_macro_f1", 4),
            ("Worst-class F1", "worst_direction_f1", 4),
            ("Trans F1 (binarized)", "transition_f1", 4),
            ("Params", "params", 0),
            ("ms/win", "inference_ms_per_window", 4),
        ], ["model", "channels"])

    # ===== Synthetic =====
    syn_csv = root / "synthetic" / "results_synthetic.csv"
    if syn_csv.exists():
        df = pd.read_csv(syn_csv)
        t = agg(df, ["task", "model"], [
            "accuracy", "macro_f1", "transition_f1", "params", "inference_ms_per_window",
        ])
        out_chunks["synthetic"] = render(t, [
            ("Acc", "accuracy", 4),
            ("**Macro F1**", "macro_f1", 4),
            ("Trans F1", "transition_f1", 4),
            ("Params", "params", 0),
            ("ms/win", "inference_ms_per_window", 4),
        ], ["task", "model"])

    # ===== Phase analysis =====
    pa_csv = root / "phase_analysis" / "all_summaries.csv"
    if pa_csv.exists():
        df = pd.read_csv(pa_csv)
        keep_cols = [c for c in df.columns if c not in ("checkpoint",)]
        out_chunks["phase_analysis"] = "| " + " | ".join(keep_cols) + " |\n| " + \
            " | ".join(["---"] * len(keep_cols)) + " |\n" + \
            "\n".join("| " + " | ".join(
                f"{r[c]:.4f}" if isinstance(r[c], float) and not np.isnan(r[c]) else str(r[c])
                for c in keep_cols
            ) + " |" for _, r in df.iterrows())

    # ===== Avg confusion matrices for direction (mamba3 + complex_ssm) =====
    cm_lines = []
    for (model, channels) in [("mamba3", "acc_gyro"), ("transformer", "acc_gyro"), ("complex_ssm", "acc_gyro")]:
        cm = average_confusion_matrix(root / "direction_5seed", model, channels)
        if cm is None:
            cm = average_confusion_matrix(root / "ssm_ablation_5seed", model, channels)
        if cm is None:
            continue
        cm_lines.append(f"\n#### {model} / {channels} — mean confusion (rows=true, cols=pred)\n")
        cm_lines.append("| true \\\\ pred | " + " | ".join(class_names) + " |")
        cm_lines.append("| --- | " + " | ".join(["---"] * len(class_names)) + " |")
        for i, cls in enumerate(class_names):
            cells = [cls] + [f"{cm[i, j]:.1f}" for j in range(cm.shape[1])]
            cm_lines.append("| " + " | ".join(cells) + " |")
    if cm_lines:
        out_chunks["confusion_matrices"] = "\n".join(cm_lines)

    Path(args.out).write_text(
        "\n\n".join(f"## {k}\n\n{v}" for k, v in out_chunks.items()), encoding="utf-8",
    )
    print(f"Wrote {args.out}")
    for k, v in out_chunks.items():
        print(f"\n### {k}\n{v}")


if __name__ == "__main__":
    main()
