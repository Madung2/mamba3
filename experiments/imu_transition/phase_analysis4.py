"""exp_plan4 §1 — re-analyse the same complex-state checkpoints with new
selective-update proxies (update_budget, forget_rate, phase_velocity).

Uses the same model class so loading existing phase-3 checkpoints works;
forward_with_state() now returns the new proxies because ssm_2x2 was extended.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

import sys
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.imu_transition.datasets.uci_har_pt import create_dataset_splits
from experiments.imu_transition.models.gyrophase import head_config_for
from experiments.imu_transition.models.phase_classifier import PhaseAwareClassifier
from experiments.imu_transition.phase_analysis3 import gyro_indices, _pearson
from experiments.imu_transition.utils import load_config, resolve_repo_path


PROXIES = ["selective_score", "forget_rate", "update_budget", "phase_velocity"]


def analyse_checkpoint(ckpt_path: Path, config: dict, channel_mode: str, device: torch.device) -> dict:
    ckpt = torch.load(ckpt_path, map_location=device)
    spec = ckpt["spec"]
    backbone_name, head_preset, pool_mode = spec
    splits = create_dataset_splits(
        data_root=resolve_repo_path(config["data_root"]),
        window_size=config["window_size"],
        stride=config["stride"],
        channel_mode=channel_mode,
        train_ratio=config["train_ratio"],
        val_ratio=config["val_ratio"],
        test_ratio=config["test_ratio"],
        seed=config["seed"],
        normalize=config.get("normalize", True),
        split_mode=config.get("split_mode", "random"),
        task=config.get("task", "binary"),
    )
    head_cfg = head_config_for(head_preset)
    model = PhaseAwareClassifier(
        backbone_name=backbone_name,
        head_config=head_cfg,
        in_channels=splits.num_channels,
        num_classes=splits.num_classes,
        gyro_indices=gyro_indices(channel_mode),
        backbone_config=None,
        dropout=0.1,
        pool=pool_mode,
    ).to(device)

    loader = DataLoader(splits.test_dataset, batch_size=64, shuffle=False)
    with torch.no_grad():
        sample_x, _ = next(iter(loader))
        sample_x = sample_x.to(device)
        _ = model(sample_x)
    try:
        model.load_state_dict(ckpt["state_dict"])
    except Exception as exc:
        return {"spec": f"{backbone_name}.{head_preset}.{pool_mode}", "seed": int(config["seed"]),
                "error": repr(exc)}
    model.eval()

    rows: list[dict] = []
    gi = gyro_indices(channel_mode)
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            _, info = model.forward_with_state(x)
            state = info["state"]
            if "h_real" not in state:
                return {"spec": f"{backbone_name}.{head_preset}.{pool_mode}", "seed": int(config["seed"])}
            h_real = state["h_real"].float()
            h_imag = state["h_imag"].float()
            phase = torch.atan2(h_imag, h_real)
            delta = torch.atan2(
                torch.sin(phase[:, 1:] - phase[:, :-1]),
                torch.cos(phase[:, 1:] - phase[:, :-1]),
            ).abs()
            dphi_mean = delta.mean(dim=(1, 2))
            if gi is None:
                gyro_mean = torch.zeros(x.shape[0], device=x.device)
            else:
                idx = torch.as_tensor(gi, device=x.device, dtype=torch.long)
                gyro_mean = torch.sqrt((x.index_select(-1, idx) ** 2).sum(dim=-1) + 1e-8).mean(dim=1)
            proxy_means = {}
            for p in PROXIES:
                if state.get(p) is not None:
                    proxy_means[p] = state[p].float().mean(dim=(1, 2))
                else:
                    proxy_means[p] = torch.zeros(x.shape[0], device=x.device)
            for i in range(x.shape[0]):
                rec = {
                    "y": int(y[i].item()),
                    "dphi": float(dphi_mean[i].item()),
                    "gyro": float(gyro_mean[i].item()),
                }
                for p in PROXIES:
                    rec[p] = float(proxy_means[p][i].item())
                rows.append(rec)
    arr_y = np.array([r["y"] for r in rows])
    arr_dphi = np.array([r["dphi"] for r in rows])
    arr_gyro = np.array([r["gyro"] for r in rows])
    is_trans = arr_y >= 1
    out = {
        "spec": f"{backbone_name}.{head_preset}.{pool_mode}",
        "seed": int(config["seed"]),
        "n_windows": len(rows),
    }
    for p in PROXIES:
        arr_p = np.array([r[p] for r in rows])
        out[f"r_{p}_gyro"] = _pearson(arr_p, arr_gyro)
        out[f"r_{p}_dphi"] = _pearson(arr_p, arr_dphi)
        if is_trans.any() and (~is_trans).any():
            mt = float(arr_p[is_trans].mean())
            mn = float(arr_p[~is_trans].mean())
            out[f"mean_{p}_trans"] = mt
            out[f"mean_{p}_non"] = mn
            out[f"ratio_{p}"] = float(mt / mn) if abs(mn) > 1e-9 else float("nan")
        else:
            out[f"ratio_{p}"] = float("nan")
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="outputs_user/imu_transition/phase3_main")
    parser.add_argument("--config", default="/tmp/phase3.yaml")
    parser.add_argument("--channels", default="acc_gyro")
    parser.add_argument("--specs-regex", default="complex_")
    args = parser.parse_args()

    base_config = load_config(args.config)
    base_config["task"] = "direction"
    base_config["split_mode"] = "random"
    base_config["channel_modes"] = [args.channels]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    root = Path(args.root)
    out: list[dict] = []
    for ckpt_path in root.rglob("best.pt"):
        if args.specs_regex not in ckpt_path.parent.name:
            continue
        seed_dir = ckpt_path.parent.parent.name
        try:
            seed = int(seed_dir.replace("seed", ""))
        except ValueError:
            continue
        cfg = dict(base_config)
        cfg["seed"] = seed
        try:
            row = analyse_checkpoint(ckpt_path, cfg, channel_mode=args.channels, device=device)
        except Exception as exc:                                       # noqa: BLE001
            print(f"  failed {ckpt_path}: {exc!r}")
            continue
        out.append(row)
        print(json.dumps({k: v for k, v in row.items() if "r_" in k or "ratio" in k or "spec" in k or "seed" in k},
                         indent=2))

    import pandas as pd
    df = pd.DataFrame(out)
    df.to_csv(root / "phase_analysis4.csv", index=False)
    agg = df.groupby("spec").mean(numeric_only=True)
    agg.to_csv(root / "phase_analysis4_summary.csv")
    print(f"\nwrote {root/'phase_analysis4.csv'}, summary:")
    print(agg.to_string())


if __name__ == "__main__":
    main()
