"""Hidden-phase / selective-score correlation analysis for phase-3.

For every checkpoint of a backbone that exposes complex hidden state
(``complex_selective`` / ``complex_static`` / the original ``complex_ssm``), we
recompute on the test set:

  - hidden_phase = atan2(h_imag, h_real)              (B, T, d_state)
  - |Δphase|     = wrap(phase[t] - phase[t-1]).abs()  (B, T-1, d_state)
  - gyro_mag     = sqrt(sum(gyro^2))                  (B, T)
  - selective_score = rho_t                            (B, T, d_state)

and reports:

  1. corr(|Δphase|_mean, gyro_mag_mean) per run
  2. corr(selective_score_mean, gyro_mag_mean) per run
  3. corr(selective_score_mean, |Δphase|_mean) per run
  4. mean(selective_score) in transition vs non-transition windows
  5. mean(selective_score) in high-gyro / high-RD subsets

This is the phase-3 extension of phase_analysis.py (which only reported #1).
"""

from __future__ import annotations

import argparse
import json
import math
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
from experiments.imu_transition.utils import ensure_repo_on_path, load_config, resolve_repo_path


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 2 or b.size < 2 or np.std(a) < 1e-9 or np.std(b) < 1e-9:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def gyro_indices(channel_mode: str) -> tuple[int, ...] | None:
    if channel_mode == "acc_gyro":
        return (3, 4, 5)
    if channel_mode == "gyro":
        return (0, 1, 2)
    return None


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
    # trigger lazy head build
    loader = DataLoader(splits.test_dataset, batch_size=64, shuffle=False)
    with torch.no_grad():
        sample_x, _ = next(iter(loader))
        sample_x = sample_x.to(device)
        _ = model(sample_x)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    rows: list[dict] = []
    gi = gyro_indices(channel_mode)
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            _, info = model.forward_with_state(x)
            state = info["state"]
            if "h_real" not in state or "h_imag" not in state:
                return {}
            h_real = state["h_real"].float()
            h_imag = state["h_imag"].float()
            phase = torch.atan2(h_imag, h_real)
            delta = torch.atan2(
                torch.sin(phase[:, 1:] - phase[:, :-1]),
                torch.cos(phase[:, 1:] - phase[:, :-1]),
            ).abs()
            dphi_mean = delta.mean(dim=(1, 2))                 # (B,)
            # gyro magnitude (using denormalised window — we cheat by using
            # the same gyro_indices on x; x is already normalised but only the
            # *ratio* matters for Pearson correlation).
            if gi is None or len(gi) == 0:
                gyro_mean = torch.zeros(x.shape[0], device=x.device)
            else:
                idx = torch.as_tensor(gi, device=x.device, dtype=torch.long)
                gyro_mag = torch.sqrt((x.index_select(-1, idx) ** 2).sum(dim=-1) + 1e-8)
                gyro_mean = gyro_mag.mean(dim=1)
            sel = state.get("selective_score")
            if sel is not None:
                sel_mean = sel.float().mean(dim=(1, 2))
            else:
                sel_mean = torch.zeros(x.shape[0], device=x.device)
            for i in range(x.shape[0]):
                rows.append({
                    "y": int(y[i].item()),
                    "dphi": float(dphi_mean[i].item()),
                    "gyro": float(gyro_mean[i].item()),
                    "sel": float(sel_mean[i].item()),
                })
    arr_y = np.array([r["y"] for r in rows])
    arr_dphi = np.array([r["dphi"] for r in rows])
    arr_gyro = np.array([r["gyro"] for r in rows])
    arr_sel = np.array([r["sel"] for r in rows])

    is_trans = arr_y >= 1
    res = {
        "spec": f"{backbone_name}.{head_preset}.{pool_mode}",
        "seed": int(config["seed"]),
        "n_windows": len(rows),
        "pearson_dphi_gyro": _pearson(arr_dphi, arr_gyro),
        "pearson_sel_gyro": _pearson(arr_sel, arr_gyro),
        "pearson_sel_dphi": _pearson(arr_sel, arr_dphi),
        "dphi_trans_mean": float(arr_dphi[is_trans].mean() if is_trans.any() else float("nan")),
        "dphi_nontrans_mean": float(arr_dphi[~is_trans].mean() if (~is_trans).any() else float("nan")),
        "dphi_ratio": float(
            (arr_dphi[is_trans].mean() / arr_dphi[~is_trans].mean())
            if (is_trans.any() and (~is_trans).any() and arr_dphi[~is_trans].mean() > 1e-9)
            else float("nan")
        ),
        "sel_trans_mean": float(arr_sel[is_trans].mean() if is_trans.any() else float("nan")),
        "sel_nontrans_mean": float(arr_sel[~is_trans].mean() if (~is_trans).any() else float("nan")),
        "sel_ratio": float(
            (arr_sel[is_trans].mean() / arr_sel[~is_trans].mean())
            if (is_trans.any() and (~is_trans).any() and arr_sel[~is_trans].mean() > 1e-9)
            else float("nan")
        ),
    }
    return res


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
        if not row:
            continue
        out.append(row)
        print(json.dumps(row, indent=2))

    if not out:
        print("no complex-state runs analysed.")
        return
    import pandas as pd
    df = pd.DataFrame(out)
    df.to_csv(root / "phase_analysis3.csv", index=False)
    agg = df.groupby("spec").mean(numeric_only=True)
    agg.to_csv(root / "phase_analysis3_summary.csv")
    print(f"\nwrote {root/'phase_analysis3.csv'} and {root/'phase_analysis3_summary.csv'}")
    print(agg.to_string())


if __name__ == "__main__":
    main()
