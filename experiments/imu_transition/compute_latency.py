"""Compute end-of-window detection latency for transition segments.

For each transition activity segment in the raw HAPT labels.txt, find the
earliest test window predicted as transition (y_pred=1). End-of-window
latency = (first_positive_window_start + window_size - segment_start) * 20 ms,
assuming 50 Hz sampling. Segments with no positive prediction are counted as
missed (excluded from latency stats but reported separately).

Usage:
    python compute_latency.py --predictions-glob 'outputs_user/.../seed*/<run>/test_predictions.json' --data-root data/uci_har_pt --output latency.csv
"""

from __future__ import annotations

import argparse
import glob
import json
from collections import defaultdict
from pathlib import Path
import sys

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.imu_transition.datasets.labels import is_transition
from experiments.imu_transition.datasets.uci_har_pt import _find_dataset_root

SAMPLE_PERIOD_MS = 20.0  # 50 Hz


def load_raw_label_segments(data_root: str) -> dict[tuple[int, int, int], list[tuple[int, int, int]]]:
    """Return {(exp_id, user_id, activity_id): [(start_idx, end_idx, segment_id), ...]}.

    start_idx / end_idx use 0-based half-open semantics matching _build_windows.
    """
    root = _find_dataset_root(data_root)
    labels_path = root / "RawData" / "labels.txt"
    rows = np.loadtxt(labels_path, dtype=np.int64)
    if rows.ndim == 1:
        rows = rows[None, :]
    segments: dict[tuple[int, int, int], list[tuple[int, int, int]]] = defaultdict(list)
    for segment_id, (exp_id, user_id, activity_id, start, end) in enumerate(rows):
        start_idx = max(int(start) - 1, 0)
        end_idx = int(end)
        segments[(int(exp_id), int(user_id), int(activity_id))].append(
            (start_idx, end_idx, segment_id)
        )
    return segments


def compute_latency_for_run(prediction_path: Path, segments_by_key: dict, data_root: str) -> dict[str, object]:
    raw = json.loads(prediction_path.read_text(encoding="utf-8"))
    window_size = int(raw["window_size"])
    y_pred = np.asarray(raw["y_pred"], dtype=np.int64)
    exp_ids = np.asarray(raw["test_exp_ids"], dtype=np.int64)
    user_ids = np.asarray(raw["test_user_ids"], dtype=np.int64)
    window_starts = np.asarray(raw["test_window_starts"], dtype=np.int64)
    activity_ids = np.asarray(raw["test_activity_ids"], dtype=np.int64)

    # Group test windows by (exp_id, user_id, activity_id) so we can match them
    # back to raw activity segments.
    grouping: dict[tuple[int, int, int], list[tuple[int, int]]] = defaultdict(list)
    for i in range(y_pred.shape[0]):
        if not is_transition(int(activity_ids[i])):
            continue
        key = (int(exp_ids[i]), int(user_ids[i]), int(activity_ids[i]))
        grouping[key].append((int(window_starts[i]), int(y_pred[i])))

    latencies_ms: list[float] = []
    n_segments_in_test = 0
    n_detected = 0
    n_missed = 0

    for key, ws_pred in grouping.items():
        candidate_segments = segments_by_key.get(key, [])
        if not candidate_segments:
            continue
        # Map each window in this key to its parent segment by checking the
        # window_start is contained in [segment.start_idx, segment.end_idx - window_size].
        windows_by_segment: dict[int, list[tuple[int, int]]] = defaultdict(list)
        for ws, yp in ws_pred:
            for seg_start, seg_end, seg_id in candidate_segments:
                if seg_start <= ws <= seg_end - window_size:
                    windows_by_segment[seg_id].append((ws, yp))
                    break

        for seg_id, items in windows_by_segment.items():
            seg_meta = next(s for s in candidate_segments if s[2] == seg_id)
            seg_start = seg_meta[0]
            n_segments_in_test += 1
            items.sort(key=lambda t: t[0])
            first_positive = next((ws for ws, yp in items if yp == 1), None)
            if first_positive is None:
                n_missed += 1
                continue
            n_detected += 1
            latency_samples = (first_positive + window_size) - seg_start
            latencies_ms.append(latency_samples * SAMPLE_PERIOD_MS)

    if latencies_ms:
        arr = np.asarray(latencies_ms, dtype=np.float64)
        latency_stats = {
            "latency_ms_mean": float(arr.mean()),
            "latency_ms_median": float(np.median(arr)),
            "latency_ms_p25": float(np.percentile(arr, 25)),
            "latency_ms_p75": float(np.percentile(arr, 75)),
        }
    else:
        latency_stats = {
            "latency_ms_mean": float("nan"),
            "latency_ms_median": float("nan"),
            "latency_ms_p25": float("nan"),
            "latency_ms_p75": float("nan"),
        }

    return {
        "model": raw["model"],
        "channels": raw["channels"],
        "seed": int(raw["seed"]),
        "window_size": window_size,
        "split_mode": raw.get("split_mode", "random"),
        "n_segments_in_test": int(n_segments_in_test),
        "n_detected": int(n_detected),
        "n_missed": int(n_missed),
        "miss_rate": float(n_missed / n_segments_in_test) if n_segments_in_test else float("nan"),
        **latency_stats,
        "prediction_path": str(prediction_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute detection latency.")
    parser.add_argument("--predictions-glob", required=True, help="Glob for test_predictions.json files.")
    parser.add_argument("--data-root", default="data/uci_har_pt", help="HAPT data root.")
    parser.add_argument("--output", required=True, help="Output CSV path.")
    args = parser.parse_args()

    segments = load_raw_label_segments(args.data_root)
    paths = sorted(Path(p) for p in glob.glob(args.predictions_glob, recursive=True))
    if not paths:
        raise SystemExit(f"No prediction files matched: {args.predictions_glob}")
    rows = [compute_latency_for_run(p, segments, args.data_root) for p in paths]
    frame = pd.DataFrame(rows)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out_path, index=False)
    print(f"Wrote latency table for {len(rows)} runs -> {out_path}")
    print(frame.to_string(index=False))


if __name__ == "__main__":
    main()
