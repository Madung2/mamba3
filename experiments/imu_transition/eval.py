"""Evaluation entry point for IMU transition detection experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.imu_transition.metrics import benchmark_inference, count_parameters
from experiments.imu_transition.models.factory import create_model
from experiments.imu_transition.train import build_dataloaders, evaluate_model, resolve_device
from experiments.imu_transition.utils import ensure_repo_on_path, load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a saved IMU transition detector checkpoint.")
    parser.add_argument("--checkpoint", required=True, help="Path to the checkpoint file.")
    parser.add_argument(
        "--config",
        default=None,
        help="Optional config path. Falls back to the config stored in the checkpoint.",
    )
    parser.add_argument("--device", default=None, help="Override device.")
    parser.add_argument("--force-rebuild", action="store_true", help="Rebuild cached windows.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    config = checkpoint.get("config", {})
    if args.config is not None:
        config = load_config(args.config)
    if args.device is not None:
        config["device"] = args.device
    if "device" not in config:
        config["device"] = "cuda"
    config["force_rebuild"] = args.force_rebuild

    device = resolve_device(config["device"], model_name=checkpoint["model_name"])
    loaders, splits = build_dataloaders(config, channel_mode=checkpoint["channel_mode"], device=device)

    model = create_model(
        model_name=checkpoint["model_name"],
        in_channels=splits.num_channels,
        num_classes=splits.num_classes,
        model_config=checkpoint["model_config"],
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"])

    criterion = torch.nn.CrossEntropyLoss(weight=splits.class_weights.to(device))
    metrics, _, _ = evaluate_model(model, loaders["test"], criterion, device)
    first_batch = next(iter(loaders["test"]))[0].to(device)
    metrics["params"] = count_parameters(model)
    metrics["inference_ms_per_window"] = benchmark_inference(
        model,
        first_batch,
        warmup_runs=config.get("timing_warmup_runs", 25),
        num_runs=config.get("timing_num_runs", 100),
    )
    metrics["checkpoint_path"] = str(Path(args.checkpoint).resolve())

    ensure_repo_on_path()
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
