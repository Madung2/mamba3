"""Run the phase-1 sweep for IMU transition detection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.imu_transition.train import train_single_experiment
from experiments.imu_transition.utils import ensure_repo_on_path, load_config, resolve_repo_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the full phase-1 IMU experiment sweep.")
    parser.add_argument(
        "--config",
        default="experiments/imu_transition/configs/phase1.yaml",
        help="Path to the YAML config file.",
    )
    parser.add_argument("--device", default=None, help="Override device from config.")
    parser.add_argument("--force-rebuild", action="store_true", help="Rebuild cached windows.")
    return parser.parse_args()


def flatten_results(results: dict[str, object]) -> dict[str, object]:
    split_sizes = results.pop("split_sizes")
    best_val_metrics = results.pop("best_val_metrics")
    flat = dict(results)
    for split_name, size in split_sizes.items():
        flat[f"{split_name}_size"] = size
    for metric_name, metric_value in best_val_metrics.items():
        flat[f"best_val_{metric_name}"] = metric_value
    return flat


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.device is not None:
        config["device"] = args.device
    config["force_rebuild"] = args.force_rebuild

    results: list[dict[str, object]] = []
    for model_name in config["models"]:
        for channel_mode in config["channel_modes"]:
            result = train_single_experiment(config=config, model_name=model_name, channel_mode=channel_mode)
            results.append(flatten_results(result))

    repo_root = ensure_repo_on_path()
    output_root = resolve_repo_path(config["output_root"], repo_root=repo_root)
    output_root.mkdir(parents=True, exist_ok=True)
    csv_path = output_root / "results_phase1.csv"
    json_path = output_root / "results_phase1.json"

    frame = pd.DataFrame(results)
    frame.to_csv(csv_path, index=False)
    json_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(frame.to_string(index=False))


if __name__ == "__main__":
    main()
