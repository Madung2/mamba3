"""Aggregate phase-3 sweep results.

Reads all per-run ``test_metrics.json`` files under
``outputs_user/imu_transition/<suffix>/seed*/<backbone>__<head>__<pool>/``
and produces:
  - ``agg_phase3.md``: mean ± std table grouped by (backbone, head).
  - ``per_run.csv``: flat per-run rows.
  - ``opposite_pair.csv``: confusion rates between paired transition classes.
  - ``rd_subset.csv``: subset Direction Macro F1 on high/low gyro / RD slices.

Designed to run after ``run_phase3.py`` finishes — but it can also run on a
partial sweep (missing runs are skipped).
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from statistics import mean, stdev

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score


PAIRS = [
    (1, 2, "stand-to-sit vs sit-to-stand"),
    (3, 4, "sit-to-lie vs lie-to-sit"),
    (5, 6, "stand-to-lie vs lie-to-stand"),
]

PRIMARY_METRICS = [
    "accuracy", "macro_f1", "direction_macro_f1", "worst_direction_f1",
    "transition_f1", "non_transition_f1", "inference_ms_per_window", "params",
]


def gather_rows(root: Path) -> list[dict]:
    rows: list[dict] = []
    for tm_path in root.rglob("test_metrics.json"):
        try:
            data = json.loads(tm_path.read_text())
        except Exception:                                              # noqa: BLE001
            continue
        # Some entries may be from non-phase3 sweeps; require backbone+head keys.
        if "backbone" not in data or "head" not in data:
            continue
        data["run_dir"] = str(tm_path.parent.relative_to(root))
        rows.append(data)
    return rows


def fmt(mean_v: float, std_v: float) -> str:
    if np.isnan(mean_v):
        return "—"
    return f"{mean_v:.4f} ± {std_v:.4f}"


def aggregate_main(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    # Build group keys
    df["model"] = df["backbone"] + " + " + df["head"]
    out: list[dict] = []
    for (model, channels, task), sub in df.groupby(["model", "channels", "task"], dropna=False):
        row = {"model": model, "channels": channels, "task": task, "n_seeds": len(sub)}
        for metric in PRIMARY_METRICS:
            vals = [v for v in sub.get(metric, pd.Series()).tolist() if isinstance(v, (int, float)) and not (isinstance(v, float) and np.isnan(v))]
            if not vals:
                row[metric] = ""
                continue
            m = mean(vals)
            s = stdev(vals) if len(vals) > 1 else 0.0
            row[metric] = fmt(m, s)
            row[f"{metric}_mean"] = m
            row[f"{metric}_std"] = s
        out.append(row)
    return pd.DataFrame(out)


def opposite_pair_table(rows: list[dict]) -> pd.DataFrame:
    out: list[dict] = []
    for row in rows:
        cm = row.get("confusion_matrix")
        if not cm:
            continue
        cm = np.array(cm)
        record = {
            "backbone": row["backbone"],
            "head": row["head"],
            "seed": row["seed"],
        }
        for a, b, name in PAIRS:
            if cm.shape[0] <= max(a, b):
                continue
            denom_a = max(1, cm[a].sum())
            denom_b = max(1, cm[b].sum())
            record[f"{name} → opp%"] = float((cm[a, b] / denom_a + cm[b, a] / denom_b) / 2)
        out.append(record)
    if not out:
        return pd.DataFrame()
    df = pd.DataFrame(out)
    pair_cols = [c for c in df.columns if "opp%" in c]
    agg = df.groupby(["backbone", "head"])[pair_cols].agg(["mean", "std"])
    return agg


def rd_subset_table(rows: list[dict], root: Path) -> pd.DataFrame:
    """Compute direction_macro_f1 on high-gyro / low-gyro / high-RD / low-RD subsets.

    Subsets are defined per-run by the test-window gyro magnitude/RD median.
    Needs the windowed cache to recompute gyro signals.
    """
    cache_dir = os.environ.get("HAPT_CACHE_DIR")
    if cache_dir is None:
        return pd.DataFrame()
    cache_path = Path(cache_dir) / "windows_128_64.npz"
    if not cache_path.exists():
        return pd.DataFrame()
    data = np.load(cache_path, allow_pickle=True)
    features_all = data["x"]                              # (N, T, 6) [acc(3), gyro(3)]
    window_starts_all = data["window_starts"]
    exp_ids_all = data["exp_ids"]
    # map (exp, start) → row index for O(1) lookup
    key_to_idx = {(int(e), int(s)): i for i, (e, s) in enumerate(zip(exp_ids_all, window_starts_all))}

    out: list[dict] = []
    for row in rows:
        run_dir = root / row["run_dir"]
        preds_path = run_dir / "test_predictions.json"
        if not preds_path.exists():
            continue
        preds = json.loads(preds_path.read_text())
        try:
            y_true = np.array(preds["y_true"], dtype=np.int64)
            y_pred = np.array(preds["y_pred"], dtype=np.int64)
            test_exp = np.array(preds["test_exp_ids"], dtype=np.int64)
            test_start = np.array(preds["test_window_starts"], dtype=np.int64)
        except KeyError:
            continue
        # gyro magnitude per window
        gyro_mag = np.zeros(len(y_true), dtype=np.float32)
        rd_std = np.zeros(len(y_true), dtype=np.float32)
        skip = False
        for i, (e, s) in enumerate(zip(test_exp, test_start)):
            idx = key_to_idx.get((int(e), int(s)))
            if idx is None:
                skip = True
                break
            window = features_all[idx]                    # (T, 6)
            gyro = window[:, 3:6]
            gyro_mag[i] = np.sqrt((gyro ** 2).sum(axis=-1)).mean()
            rd_std[i] = gyro.std(axis=0).sum()
        if skip:
            continue

        record = {
            "backbone": row["backbone"],
            "head": row["head"],
            "seed": row["seed"],
        }
        for col, signal in [("gyro", gyro_mag), ("rd", rd_std)]:
            median = float(np.median(signal))
            high_mask = signal >= median
            low_mask = ~high_mask
            for slc_name, mask in (("high", high_mask), ("low", low_mask)):
                if mask.sum() == 0:
                    record[f"{col}_{slc_name}_dirF1"] = float("nan")
                    continue
                yt = y_true[mask]
                yp = y_pred[mask]
                # direction macro f1 on classes 1..6
                f1 = f1_score(yt, yp, labels=list(range(1, 7)), average="macro", zero_division=0)
                record[f"{col}_{slc_name}_dirF1"] = float(f1)
        # high-gyro AND high-RD
        hg_mask = gyro_mag >= np.median(gyro_mag)
        hr_mask = rd_std >= np.median(rd_std)
        cross = hg_mask & hr_mask
        if cross.sum() > 0:
            record["gyro_high_rd_high_dirF1"] = float(
                f1_score(y_true[cross], y_pred[cross], labels=list(range(1, 7)),
                         average="macro", zero_division=0)
            )
        out.append(record)
    if not out:
        return pd.DataFrame()
    df = pd.DataFrame(out)
    agg = df.groupby(["backbone", "head"]).mean(numeric_only=True)
    return agg


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="outputs_user/imu_transition/phase3_main")
    args = parser.parse_args()
    root = Path(args.root)

    rows = gather_rows(root)
    if not rows:
        print(f"No phase-3 runs found under {root}")
        return

    per_run = pd.DataFrame(rows)
    per_run.to_csv(root / "per_run.csv", index=False)

    main_table = aggregate_main(rows)
    md_path = root / "agg_phase3.md"
    with md_path.open("w") as f:
        f.write("# Phase-3 sweep aggregate\n\n")
        f.write(f"runs: {len(rows)}\n\n")
        cols = ["model", "channels", "task", "n_seeds",
                "direction_macro_f1", "macro_f1", "transition_f1",
                "non_transition_f1", "worst_direction_f1",
                "accuracy", "inference_ms_per_window", "params"]
        present = [c for c in cols if c in main_table.columns]
        f.write(main_table[present].to_markdown(index=False))
        f.write("\n")
    print(f"wrote {md_path}")

    opp = opposite_pair_table(rows)
    if not opp.empty:
        opp.to_csv(root / "opposite_pair.csv")
        print(f"wrote {root/'opposite_pair.csv'}")

    rd = rd_subset_table(rows, root)
    if not rd.empty:
        rd.to_csv(root / "rd_subset.csv")
        print(f"wrote {root/'rd_subset.csv'}")


if __name__ == "__main__":
    main()
