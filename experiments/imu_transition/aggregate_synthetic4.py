"""Aggregate synthetic4 results into mean±std per (backbone, task)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, stdev

import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="outputs_user/imu_transition/synthetic4")
    args = parser.parse_args()
    root = Path(args.root)

    rows = json.loads((root / "results_synthetic4.json").read_text())
    rows = [r for r in rows if "error" not in r]
    df = pd.DataFrame(rows)
    out = []
    for (backbone, task), sub in df.groupby(["backbone", "task"]):
        row = {"backbone": backbone, "task": task, "n_seeds": len(sub)}
        for col in ["macro_f1", "accuracy", "worst_class_f1", "inference_ms_per_window", "params"]:
            vals = sub[col].dropna().astype(float).tolist()
            if not vals:
                row[col] = "—"
                continue
            m, s = mean(vals), stdev(vals) if len(vals) > 1 else 0.0
            row[col] = f"{m:.4f} ± {s:.4f}"
            row[f"{col}_mean"] = m
            row[f"{col}_std"] = s
        out.append(row)
    agg = pd.DataFrame(out).sort_values(["task", "macro_f1_mean"], ascending=[True, False])
    agg.to_csv(root / "agg_synthetic4.csv", index=False)
    md_path = root / "agg_synthetic4.md"
    with md_path.open("w") as f:
        f.write("# synthetic4 aggregate\n\n")
        for task, sub in agg.groupby("task"):
            f.write(f"## Task: {task}\n\n")
            cols = ["backbone", "macro_f1", "worst_class_f1", "accuracy", "inference_ms_per_window", "params"]
            present = [c for c in cols if c in sub.columns]
            f.write(sub[present].to_markdown(index=False))
            f.write("\n\n")
    print(f"wrote {md_path}\n")
    print(open(md_path).read())


if __name__ == "__main__":
    main()
