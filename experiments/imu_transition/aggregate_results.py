"""Aggregate sweep CSVs into mean ± std tables for result.md."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

JOIN_KEYS = ["model", "channels", "seed", "window_size", "split_mode"]


def load_sweep(sweep_dir: Path) -> pd.DataFrame:
    r = pd.read_csv(sweep_dir / "results_phase1.csv")
    l = pd.read_csv(sweep_dir / "latency.csv")
    df = r.merge(l, on=JOIN_KEYS, how="inner", validate="one_to_one")
    return df


def agg_table(df: pd.DataFrame, group: list[str], metrics: list[str]) -> pd.DataFrame:
    g = df.groupby(group, sort=True)
    rows = []
    for key, sub in g:
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


def fmt_meanstd(mean: float, std: float, digits: int = 4) -> str:
    if np.isnan(mean):
        return "nan"
    return f"{mean:.{digits}f} ± {std:.{digits}f}"


def render_markdown_table(df: pd.DataFrame, columns: list[tuple[str, str, int]], group_cols: list[str]) -> str:
    """columns: list of (display_label, source_metric_name, digits)."""
    lines = []
    header = " | ".join(group_cols + [c[0] for c in columns])
    sep = " | ".join(["---"] * (len(group_cols) + len(columns)))
    lines.append("| " + header + " |")
    lines.append("| " + sep + " |")
    for _, r in df.iterrows():
        cells = [str(r[c]) for c in group_cols]
        for label, metric, digits in columns:
            cells.append(fmt_meanstd(r[f"{metric}_mean"], r[f"{metric}_std"], digits))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="outputs_user/imu_transition")
    parser.add_argument("--summary-json", default="outputs_user/imu_transition/summary.json")
    args = parser.parse_args()

    root = Path(args.output_dir)
    sweeps = {name: load_sweep(root / name) for name in (
        "main_5seed", "subject_5seed", "window_64_3seed",
        "window_128_3seed", "window_256_3seed",
    )}

    metric_set = [
        "accuracy", "macro_f1", "transition_precision", "transition_recall",
        "transition_f1", "params", "inference_ms_per_window",
        "latency_ms_mean", "latency_ms_median", "miss_rate",
    ]

    out: dict[str, str] = {}

    # Main 5-seed: per (model, channels)
    df = sweeps["main_5seed"]
    table_main = agg_table(df, ["model", "channels"], metric_set)
    out["main_by_model_channels"] = render_markdown_table(
        table_main,
        [
            ("Acc", "accuracy", 4),
            ("Macro F1", "macro_f1", 4),
            ("Trans P", "transition_precision", 4),
            ("Trans R", "transition_recall", 4),
            ("**Trans F1**", "transition_f1", 4),
            ("Latency ms", "latency_ms_mean", 1),
            ("Miss rate", "miss_rate", 4),
            ("Infer ms/win", "inference_ms_per_window", 4),
            ("Params", "params", 0),
        ],
        ["model", "channels"],
    )

    # Main acc_gyro only - the paper headline
    headline = table_main[table_main["channels"] == "acc_gyro"].copy()
    out["main_headline_acc_gyro"] = render_markdown_table(
        headline,
        [
            ("Acc", "accuracy", 4),
            ("Macro F1", "macro_f1", 4),
            ("Trans P", "transition_precision", 4),
            ("Trans R", "transition_recall", 4),
            ("**Trans F1**", "transition_f1", 4),
            ("Latency ms", "latency_ms_mean", 1),
            ("Miss rate", "miss_rate", 4),
            ("Infer ms/win", "inference_ms_per_window", 4),
            ("Params", "params", 0),
        ],
        ["model"],
    )

    # Subject 5-seed: per (model, channels)
    df = sweeps["subject_5seed"]
    table_subj = agg_table(df, ["model", "channels"], metric_set)
    out["subject_by_model_channels"] = render_markdown_table(
        table_subj,
        [
            ("Acc", "accuracy", 4),
            ("Macro F1", "macro_f1", 4),
            ("Trans P", "transition_precision", 4),
            ("Trans R", "transition_recall", 4),
            ("**Trans F1**", "transition_f1", 4),
            ("Latency ms", "latency_ms_mean", 1),
            ("Miss rate", "miss_rate", 4),
            ("Infer ms/win", "inference_ms_per_window", 4),
        ],
        ["model", "channels"],
    )
    subj_headline = table_subj[table_subj["channels"] == "acc_gyro"].copy()
    out["subject_headline_acc_gyro"] = render_markdown_table(
        subj_headline,
        [
            ("Acc", "accuracy", 4),
            ("Macro F1", "macro_f1", 4),
            ("Trans P", "transition_precision", 4),
            ("Trans R", "transition_recall", 4),
            ("**Trans F1**", "transition_f1", 4),
            ("Latency ms", "latency_ms_mean", 1),
            ("Miss rate", "miss_rate", 4),
        ],
        ["model"],
    )

    # Random vs subject delta on acc_gyro
    delta_rows = []
    for m in headline["model"]:
        rand_row = headline[headline["model"] == m].iloc[0]
        subj_row = subj_headline[subj_headline["model"] == m].iloc[0]
        delta_rows.append({
            "model": m,
            "random_t_f1": fmt_meanstd(rand_row["transition_f1_mean"], rand_row["transition_f1_std"]),
            "subject_t_f1": fmt_meanstd(subj_row["transition_f1_mean"], subj_row["transition_f1_std"]),
            "delta_t_f1": f"{subj_row['transition_f1_mean'] - rand_row['transition_f1_mean']:+.4f}",
            "random_acc": fmt_meanstd(rand_row["accuracy_mean"], rand_row["accuracy_std"]),
            "subject_acc": fmt_meanstd(subj_row["accuracy_mean"], subj_row["accuracy_std"]),
        })
    out["random_vs_subject"] = "\n".join([
        "| Model | Random split T-F1 | Subject-indep T-F1 | Δ T-F1 | Random Acc | Subject Acc |",
        "| --- | --- | --- | --- | --- | --- |",
        *[
            f"| {r['model']} | {r['random_t_f1']} | {r['subject_t_f1']} | {r['delta_t_f1']} | {r['random_acc']} | {r['subject_acc']} |"
            for r in delta_rows
        ],
    ])

    # Window ablation: aggregate three sweeps
    wdfs = []
    for w in (64, 128, 256):
        d = sweeps[f"window_{w}_3seed"].copy()
        d["window_size"] = w
        wdfs.append(d)
    wdf = pd.concat(wdfs, ignore_index=True)
    table_win = agg_table(wdf, ["model", "window_size"], metric_set)
    out["window_ablation"] = render_markdown_table(
        table_win,
        [
            ("Acc", "accuracy", 4),
            ("Macro F1", "macro_f1", 4),
            ("Trans P", "transition_precision", 4),
            ("Trans R", "transition_recall", 4),
            ("**Trans F1**", "transition_f1", 4),
            ("Latency ms", "latency_ms_mean", 1),
            ("Miss rate", "miss_rate", 4),
            ("Infer ms/win", "inference_ms_per_window", 4),
        ],
        ["model", "window_size"],
    )

    # Save flat JSON summary too (mean+std of every metric)
    summary = {
        "main_5seed": table_main.to_dict(orient="records"),
        "subject_5seed": table_subj.to_dict(orient="records"),
        "window_ablation": table_win.to_dict(orient="records"),
    }
    Path(args.summary_json).write_text(json.dumps(summary, indent=2, default=float))

    # Dump all markdown tables to a single file for easy inclusion
    md_path = Path(args.output_dir) / "agg_tables.md"
    with md_path.open("w") as f:
        for key, table in out.items():
            f.write(f"## {key}\n\n{table}\n\n")
    print(f"Wrote {md_path}")
    print(f"Wrote {args.summary_json}")
    print()
    for key, table in out.items():
        print(f"### {key}")
        print(table)
        print()


if __name__ == "__main__":
    main()
