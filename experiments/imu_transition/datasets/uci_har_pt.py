"""UCI HAPT data loading, windowing, and random splitting."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

from .labels import ACTIVITY_LABELS, is_transition, resolve_channel_indices


class WindowDataset(Dataset):
    """Simple in-memory dataset for fixed-length IMU windows."""

    def __init__(self, features: np.ndarray, labels: np.ndarray):
        self.features = np.ascontiguousarray(features.astype(np.float32))
        self.labels = np.ascontiguousarray(labels.astype(np.int64))

    def __len__(self) -> int:
        return int(self.labels.shape[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.from_numpy(self.features[index]), torch.tensor(self.labels[index], dtype=torch.long)


@dataclass
class DatasetSplits:
    train_dataset: WindowDataset
    val_dataset: WindowDataset
    test_dataset: WindowDataset
    class_weights: torch.Tensor
    num_channels: int
    num_classes: int
    split_sizes: dict[str, int]
    normalization: dict[str, np.ndarray]
    metadata: dict[str, Any]


def _stratify_or_none(labels: np.ndarray) -> np.ndarray | None:
    return labels if np.unique(labels).size > 1 else None


def _resolve_data_root(data_root: str | Path) -> Path:
    return Path(data_root).expanduser().resolve()


def _find_dataset_root(data_root: str | Path) -> Path:
    root = _resolve_data_root(data_root)
    candidates = [
        root,
        root / "HAPT Data Set",
        root / "extracted",
        root / "dataset",
    ]
    for candidate in candidates:
        if (candidate / "RawData" / "labels.txt").exists():
            return candidate
    for label_path in root.rglob("labels.txt"):
        if label_path.parent.name == "RawData":
            return label_path.parent.parent
    raise FileNotFoundError(
        f"Could not find UCI HAPT raw data under '{root}'. "
        "Run scripts/download_uci_har_pt.sh first."
    )


def _cache_path(dataset_root: Path, window_size: int, stride: int) -> Path:
    return dataset_root.parent / f"windows_{window_size}_{stride}.npz"


def _load_signal_pair(raw_data_dir: Path, exp_id: int, user_id: int) -> np.ndarray:
    acc_path = raw_data_dir / f"acc_exp{exp_id:02d}_user{user_id:02d}.txt"
    gyro_path = raw_data_dir / f"gyro_exp{exp_id:02d}_user{user_id:02d}.txt"
    if not acc_path.exists() or not gyro_path.exists():
        raise FileNotFoundError(f"Missing raw signal files for exp={exp_id}, user={user_id}.")
    acc = np.loadtxt(acc_path, dtype=np.float32)
    gyro = np.loadtxt(gyro_path, dtype=np.float32)
    if acc.ndim != 2 or gyro.ndim != 2 or acc.shape[1] != 3 or gyro.shape[1] != 3:
        raise ValueError(f"Unexpected raw signal shape for exp={exp_id}, user={user_id}.")
    return np.concatenate([acc, gyro], axis=1)


def _build_windows(dataset_root: Path, window_size: int, stride: int) -> dict[str, np.ndarray]:
    raw_data_dir = dataset_root / "RawData"
    labels_path = raw_data_dir / "labels.txt"
    labels = np.loadtxt(labels_path, dtype=np.int64)
    if labels.ndim == 1:
        labels = labels[None, :]

    signal_cache: dict[tuple[int, int], np.ndarray] = {}
    features: list[np.ndarray] = []
    binary_labels: list[int] = []
    activity_ids: list[int] = []
    exp_ids: list[int] = []
    user_ids: list[int] = []
    start_indices: list[int] = []

    for exp_id, user_id, activity_id, start, end in labels:
        signal_key = (int(exp_id), int(user_id))
        if signal_key not in signal_cache:
            signal_cache[signal_key] = _load_signal_pair(raw_data_dir, *signal_key)
        signal = signal_cache[signal_key]

        start_idx = max(int(start) - 1, 0)
        end_idx = min(int(end), signal.shape[0])
        if end_idx - start_idx < window_size:
            continue

        for window_start in range(start_idx, end_idx - window_size + 1, stride):
            window_end = window_start + window_size
            features.append(signal[window_start:window_end])
            binary_labels.append(is_transition(int(activity_id)))
            activity_ids.append(int(activity_id))
            exp_ids.append(int(exp_id))
            user_ids.append(int(user_id))
            start_indices.append(int(window_start))

    if not features:
        raise RuntimeError("No windows were generated from the UCI HAPT raw labels.")

    x = np.stack(features).astype(np.float32)
    y_binary = np.asarray(binary_labels, dtype=np.int64)
    y_activity = np.asarray(activity_ids, dtype=np.int64)
    exp_arr = np.asarray(exp_ids, dtype=np.int64)
    user_arr = np.asarray(user_ids, dtype=np.int64)
    start_arr = np.asarray(start_indices, dtype=np.int64)

    return {
        "x": x,
        "y_binary": y_binary,
        "y_activity": y_activity,
        "exp_ids": exp_arr,
        "user_ids": user_arr,
        "window_starts": start_arr,
    }


def load_or_create_windows(
    data_root: str | Path,
    window_size: int,
    stride: int,
    force_rebuild: bool = False,
) -> dict[str, np.ndarray]:
    dataset_root = _find_dataset_root(data_root)
    cache_path = _cache_path(dataset_root, window_size, stride)
    if cache_path.exists() and not force_rebuild:
        cached = np.load(cache_path, allow_pickle=False)
        return {key: cached[key] for key in cached.files}

    arrays = _build_windows(dataset_root, window_size=window_size, stride=stride)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, **arrays)
    return arrays


def _take_subset(indices: np.ndarray, labels: np.ndarray, subset_fraction: float, seed: int) -> np.ndarray:
    if subset_fraction >= 1.0:
        return indices
    if subset_fraction <= 0.0:
        raise ValueError("subset_fraction must be in the range (0, 1].")
    subset_size = max(2, int(len(indices) * subset_fraction))
    subset_size = min(subset_size, len(indices))
    subset, _ = train_test_split(
        indices,
        train_size=subset_size,
        random_state=seed,
        stratify=_stratify_or_none(labels[indices]),
    )
    return np.sort(subset)


def _normalize_splits(
    x_train: np.ndarray,
    x_val: np.ndarray,
    x_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    mean = x_train.mean(axis=(0, 1), keepdims=True).astype(np.float32)
    std = x_train.std(axis=(0, 1), keepdims=True).astype(np.float32)
    std = np.clip(std, a_min=1e-6, a_max=None)
    return (
        ((x_train - mean) / std).astype(np.float32),
        ((x_val - mean) / std).astype(np.float32),
        ((x_test - mean) / std).astype(np.float32),
        {"mean": mean.squeeze(0), "std": std.squeeze(0)},
    )


def create_dataset_splits(
    data_root: str | Path,
    window_size: int,
    stride: int,
    channel_mode: str,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
    normalize: bool = True,
    subset_fraction: float = 1.0,
    force_rebuild: bool = False,
) -> DatasetSplits:
    if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
        raise ValueError("train_ratio + val_ratio + test_ratio must sum to 1.0.")

    arrays = load_or_create_windows(
        data_root=data_root,
        window_size=window_size,
        stride=stride,
        force_rebuild=force_rebuild,
    )
    channel_indices = np.asarray(resolve_channel_indices(channel_mode), dtype=np.int64)

    x = arrays["x"][:, :, channel_indices]
    y = arrays["y_binary"]
    activity_ids = arrays["y_activity"]
    user_ids = arrays["user_ids"]
    exp_ids = arrays["exp_ids"]
    window_starts = arrays["window_starts"]
    indices = np.arange(y.shape[0], dtype=np.int64)

    indices = _take_subset(indices, y, subset_fraction=subset_fraction, seed=seed)
    x = x[indices]
    y = y[indices]
    activity_ids = activity_ids[indices]
    user_ids = user_ids[indices]
    exp_ids = exp_ids[indices]
    window_starts = window_starts[indices]

    train_idx, temp_idx = train_test_split(
        np.arange(y.shape[0]),
        train_size=train_ratio,
        random_state=seed,
        stratify=_stratify_or_none(y),
    )
    relative_test_ratio = test_ratio / (val_ratio + test_ratio)
    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=relative_test_ratio,
        random_state=seed,
        stratify=_stratify_or_none(y[temp_idx]),
    )

    x_train = x[train_idx]
    x_val = x[val_idx]
    x_test = x[test_idx]

    if normalize:
        x_train, x_val, x_test, normalization = _normalize_splits(x_train, x_val, x_test)
    else:
        normalization = {
            "mean": np.zeros((1, x.shape[-1]), dtype=np.float32),
            "std": np.ones((1, x.shape[-1]), dtype=np.float32),
        }

    y_train = y[train_idx]
    y_val = y[val_idx]
    y_test = y[test_idx]

    counts = np.bincount(y_train, minlength=2).astype(np.float32)
    counts = np.clip(counts, a_min=1.0, a_max=None)
    class_weights = torch.tensor(y_train.shape[0] / (2.0 * counts), dtype=torch.float32)

    metadata = {
        "activity_label_map": ACTIVITY_LABELS,
        "train_activity_ids": activity_ids[train_idx],
        "val_activity_ids": activity_ids[val_idx],
        "test_activity_ids": activity_ids[test_idx],
        "train_user_ids": user_ids[train_idx],
        "val_user_ids": user_ids[val_idx],
        "test_user_ids": user_ids[test_idx],
        "train_exp_ids": exp_ids[train_idx],
        "val_exp_ids": exp_ids[val_idx],
        "test_exp_ids": exp_ids[test_idx],
        "train_window_starts": window_starts[train_idx],
        "val_window_starts": window_starts[val_idx],
        "test_window_starts": window_starts[test_idx],
    }

    return DatasetSplits(
        train_dataset=WindowDataset(x_train, y_train),
        val_dataset=WindowDataset(x_val, y_val),
        test_dataset=WindowDataset(x_test, y_test),
        class_weights=class_weights,
        num_channels=x.shape[-1],
        num_classes=2,
        split_sizes={
            "train": int(y_train.shape[0]),
            "val": int(y_val.shape[0]),
            "test": int(y_test.shape[0]),
        },
        normalization=normalization,
        metadata=metadata,
    )
