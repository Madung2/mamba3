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
    parser.add_argument("--seeds", nargs="*", type=int, default=None, help="List of seeds; overrides config seed.")
    parser.add_argument("--split-mode", default=None, choices=["random", "subject"], help="Override split mode.")
    parser.add_argument("--task", default=None, choices=["binary", "direction"], help="Override classification task.")
    parser.add_argument("--early-stop-metric", default=None, help="Override early-stop metric.")
    parser.add_argument("--window-size", type=int, default=None, help="Override window size.")
    parser.add_argument("--stride", type=int, default=None, help="Override stride.")
    parser.add_argument("--models", nargs="*", default=None, help="Subset of models to run.")
    parser.add_argument("--channels", nargs="*", default=None, help="Subset of channel modes.")
    parser.add_argument("--output-suffix", default=None, help="Append to output_root and CSV file names.")
    parser.add_argument("--output-root", default=None, help="Override output_root entirely.")
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
    if args.split_mode is not None:
        config["split_mode"] = args.split_mode
    if args.task is not None:
        config["task"] = args.task
        if args.early_stop_metric is None and args.task == "direction":
            config["early_stop_metric"] = "direction_macro_f1"
    if args.early_stop_metric is not None:
        config["early_stop_metric"] = args.early_stop_metric
    if args.window_size is not None:
        config["window_size"] = args.window_size
    if args.stride is not None:
        config["stride"] = args.stride
    if args.models is not None:
        config["models"] = args.models
    if args.channels is not None:
        config["channel_modes"] = args.channels
    if args.output_root is not None:
        config["output_root"] = args.output_root
    if args.output_suffix:
        config["output_root"] = str(Path(config["output_root"]) / args.output_suffix)
    config["force_rebuild"] = args.force_rebuild

    seeds = args.seeds if args.seeds else [config["seed"]]

    repo_root = ensure_repo_on_path()
    output_root = resolve_repo_path(config["output_root"], repo_root=repo_root)
    output_root.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, object]] = []
    for seed in seeds:
        for model_name in config["models"]:
            for channel_mode in config["channel_modes"]:
                cfg = dict(config)
                cfg["seed"] = int(seed)
                cfg["_run_subdir"] = f"seed{seed}_{model_name}_{channel_mode}"
                # train_single_experiment uses build_run_name(model, channels) for the
                # run_dir name, which would collide across seeds. We side-step by
                # rerooting output_root per seed.
                cfg["output_root"] = str(output_root / f"seed{seed}")
                Path(cfg["output_root"]).mkdir(parents=True, exist_ok=True)
                result = train_single_experiment(
                    config=cfg, model_name=model_name, channel_mode=channel_mode
                )
                row = flatten_results(result)
                row["seed"] = int(seed)
                row.setdefault("split_mode", cfg.get("split_mode", "random"))
                row.setdefault("window_size", cfg["window_size"])
                row.setdefault("stride", cfg["stride"])
                results.append(row)

    csv_path = output_root / "results_phase1.csv"
    json_path = output_root / "results_phase1.json"

    frame = pd.DataFrame(results)
    frame.to_csv(csv_path, index=False)
    json_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\n=== Sweep finished. Wrote {csv_path} ({len(results)} rows) ===")
    print(frame.to_string(index=False))


if __name__ == "__main__":
    main()
