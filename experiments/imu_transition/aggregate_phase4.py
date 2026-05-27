"""exp_plan4 §3 — transition-only subset analysis.

Restricts to test windows where y_true >= 1 (any direction class), then splits
by gyro magnitude / RD_std median *within that subset*. Computes
direction macro F1 on each half.

Also computes the same metric on the full test set for reference.
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


def _macro_dir_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(
        f1_score(y_true, y_pred, labels=list(range(1, 7)), average="macro", zero_division=0)
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="outputs_user/imu_transition/phase3_main")
    args = parser.parse_args()
    root = Path(args.root)

    cache_dir = os.environ.get("HAPT_CACHE_DIR")
    if cache_dir is None:
        raise RuntimeError("HAPT_CACHE_DIR must be set")
    data = np.load(Path(cache_dir) / "windows_128_64.npz", allow_pickle=True)
    features_all = data["x"]
    starts_all = data["window_starts"]
    exps_all = data["exp_ids"]
    key_to_idx = {(int(e), int(s)): i for i, (e, s) in enumerate(zip(exps_all, starts_all))}

    rows: list[dict] = []
    for tm_path in root.rglob("test_metrics.json"):
        data = json.loads(tm_path.read_text())
        if "backbone" not in data:
            continue
        run_dir = tm_path.parent
        pred_path = run_dir / "test_predictions.json"
        if not pred_path.exists():
            continue
        preds = json.loads(pred_path.read_text())
        y_true = np.array(preds["y_true"], dtype=np.int64)
        y_pred = np.array(preds["y_pred"], dtype=np.int64)
        test_exp = np.array(preds["test_exp_ids"], dtype=np.int64)
        test_start = np.array(preds["test_window_starts"], dtype=np.int64)

        gyro_mag = np.zeros(len(y_true), dtype=np.float32)
        rd_std = np.zeros(len(y_true), dtype=np.float32)
        skip = False
        for i, (e, s) in enumerate(zip(test_exp, test_start)):
            idx = key_to_idx.get((int(e), int(s)))
            if idx is None:
                skip = True
                break
            window = features_all[idx]
            gyro = window[:, 3:6]
            gyro_mag[i] = np.sqrt((gyro ** 2).sum(axis=-1)).mean()
            rd_std[i] = gyro.std(axis=0).sum()
        if skip:
            continue

        is_trans = y_true >= 1
        if is_trans.sum() == 0:
            continue
        row = {
            "backbone": data["backbone"],
            "head": data["head"],
            "seed": data["seed"],
            "n_trans": int(is_trans.sum()),
            "full_dirF1": _macro_dir_f1(y_true, y_pred),
            "trans_only_dirF1": _macro_dir_f1(y_true[is_trans], y_pred[is_trans]),
        }
        trans_gyro = gyro_mag[is_trans]
        trans_rd = rd_std[is_trans]
        med_g = float(np.median(trans_gyro))
        med_r = float(np.median(trans_rd))
        for sig_name, sig, med in (("gyro", trans_gyro, med_g), ("rd", trans_rd, med_r)):
            hi = sig >= med
            lo = ~hi
            yt_h = y_true[is_trans][hi]
            yp_h = y_pred[is_trans][hi]
            yt_l = y_true[is_trans][lo]
            yp_l = y_pred[is_trans][lo]
            row[f"trans_{sig_name}_high_dirF1"] = _macro_dir_f1(yt_h, yp_h) if yt_h.size > 0 else float("nan")
            row[f"trans_{sig_name}_low_dirF1"] = _macro_dir_f1(yt_l, yp_l) if yt_l.size > 0 else float("nan")
        cross_hi = (trans_gyro >= med_g) & (trans_rd >= med_r)
        if cross_hi.any():
            row["trans_gyrohi_rdhi_dirF1"] = _macro_dir_f1(
                y_true[is_trans][cross_hi], y_pred[is_trans][cross_hi]
            )
        else:
            row["trans_gyrohi_rdhi_dirF1"] = float("nan")
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(root / "transition_only_subset.csv", index=False)
    print(f"wrote {root/'transition_only_subset.csv'} ({len(df)} rows)")

    # Aggregate by model
    grp = df.groupby(["backbone", "head"])
    metric_cols = [c for c in df.columns if c.endswith("dirF1")]
    agg = grp[metric_cols].agg(["mean", "std"])
    agg.to_csv(root / "transition_only_subset_agg.csv")
    print(f"wrote {root/'transition_only_subset_agg.csv'}")
    print(agg.to_string())


if __name__ == "__main__":
    main()
