"""GyroPhase Head and Selective GyroPhase Head.

Builds a window-level phase-aware representation from
- hidden magnitude / phase change (if the encoder exposes complex state),
- input gyro magnitude,
- Rotation Diversity (window-level),
- selective_score / update strength (if available),

and concatenates it with the encoder's base representation before the
classifier. The feature set is controlled by ``HeadConfig`` so we can run
strict ablations (Magnitude only / Phase only / GyroPhase / GyroPhase+RD /
Selective GyroPhase) with one code path.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math

import torch
from torch import nn


def wrap_to_pi(delta: torch.Tensor) -> torch.Tensor:
    """Wrap angle differences into (-pi, pi]."""
    return torch.atan2(torch.sin(delta), torch.cos(delta))


def compute_phase_change(h_real: torch.Tensor, h_imag: torch.Tensor) -> torch.Tensor:
    """|Δphase_t| over the time axis, shape (B, T-1, d_state)."""
    phase = torch.atan2(h_imag.float(), h_real.float())
    delta = wrap_to_pi(phase[:, 1:] - phase[:, :-1])
    return delta.abs()


def compute_gyro_magnitude(x: torch.Tensor, gyro_indices: tuple[int, ...] | None) -> torch.Tensor:
    """gyro_mag shape (B, T) — zeros if gyro channels are absent."""
    if gyro_indices is None or len(gyro_indices) == 0:
        return torch.zeros(x.size(0), x.size(1), device=x.device, dtype=x.dtype)
    idx = torch.as_tensor(gyro_indices, device=x.device, dtype=torch.long)
    gyro = x.index_select(-1, idx)
    return torch.sqrt((gyro ** 2).sum(dim=-1) + 1e-8)


def compute_rotation_diversity_std(
    x: torch.Tensor, gyro_indices: tuple[int, ...] | None
) -> torch.Tensor:
    """RD_std = std(gyro_x)+std(gyro_y)+std(gyro_z) per window — shape (B,)."""
    if gyro_indices is None or len(gyro_indices) == 0:
        return torch.zeros(x.size(0), device=x.device, dtype=x.dtype)
    idx = torch.as_tensor(gyro_indices, device=x.device, dtype=torch.long)
    gyro = x.index_select(-1, idx)
    return gyro.std(dim=1).sum(dim=-1)


def compute_rotation_diversity_bin(
    x: torch.Tensor, gyro_indices: tuple[int, ...] | None, n_bins: int = 6
) -> torch.Tensor:
    """Count visited gyro-direction bins per window — shape (B,).

    Direction is the unit-vector of the (gyro_x, gyro_y, gyro_z) sample. We
    quantise (theta, phi) into ``n_bins`` × ``n_bins`` cells and count unique
    cells per window. Differentiable proxy: we do NOT backprop through this.
    """
    if gyro_indices is None or len(gyro_indices) == 0 or len(gyro_indices) < 3:
        return torch.zeros(x.size(0), device=x.device, dtype=x.dtype)
    idx = torch.as_tensor(gyro_indices, device=x.device, dtype=torch.long)
    gyro = x.index_select(-1, idx).detach()
    norm = gyro.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    unit = gyro / norm
    # spherical coords
    theta = torch.acos(unit[..., 2].clamp(-1.0, 1.0))               # (B, T) in [0, pi]
    phi = torch.atan2(unit[..., 1], unit[..., 0])                    # (B, T) in (-pi, pi]
    theta_bin = (theta / math.pi * n_bins).clamp(max=n_bins - 1).long()
    phi_bin = ((phi + math.pi) / (2 * math.pi) * n_bins).clamp(max=n_bins - 1).long()
    code = theta_bin * n_bins + phi_bin                              # (B, T) in [0, n_bins^2)
    out = torch.zeros(x.size(0), device=x.device, dtype=x.dtype)
    for i in range(x.size(0)):
        out[i] = code[i].unique().numel()
    return out


@dataclass
class HeadConfig:
    """Toggleable feature flags for GyroPhase variants."""

    use_base: bool = True                       # use encoder pooled output
    use_hidden_magnitude: bool = False
    use_hidden_phase: bool = False
    use_gyro_magnitude: bool = False
    use_rotation_diversity: bool = False
    use_selective_score: bool = False
    use_interactions: bool = False              # p*q, p*d, q*d, p*q*d, etc.
    rd_kind: str = "std"                        # "std" or "bin"
    pooling: tuple[str, ...] = ("mean", "max", "std")
    # Which state-dict key to map to `selective_score`. Defaults to legacy
    # rho-based proxy ("selective_score"). exp_plan4 §1 adds
    # {"update_budget", "forget_rate", "phase_velocity"}.
    selective_proxy: str = "selective_score"


def _pool_timeseries(x: torch.Tensor, kinds: tuple[str, ...]) -> list[torch.Tensor]:
    """Pool a (B, T) or (B, T, C) tensor along T into a list of (B, *) tensors."""
    pieces: list[torch.Tensor] = []
    if x.dim() == 2:
        x = x.unsqueeze(-1)  # (B, T, 1)
    for kind in kinds:
        if kind == "mean":
            pieces.append(x.mean(dim=1))
        elif kind == "max":
            pieces.append(x.max(dim=1).values)
        elif kind == "std":
            pieces.append(x.std(dim=1))
        else:
            raise ValueError(f"Unknown pool kind '{kind}'.")
    return pieces


class GyroPhaseHead(nn.Module):
    """Concat encoder base feature with phase-aware features and classify.

    The feature vector is built lazily on first forward to determine its size;
    once known we register the classifier (linear). This avoids hard-coding
    feature dims per config combo.
    """

    def __init__(
        self,
        d_base: int,
        num_classes: int,
        config: HeadConfig,
        hidden_dim: int | None = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_base = d_base
        self.num_classes = num_classes
        self.config = config
        self.hidden_dim = hidden_dim
        self.dropout = nn.Dropout(dropout)
        self.classifier: nn.Module | None = None  # built on first forward
        self._feature_dim: int | None = None

    def _build_classifier(self, feature_dim: int, device: torch.device) -> None:
        self._feature_dim = feature_dim
        if self.hidden_dim is None:
            self.classifier = nn.Linear(feature_dim, self.num_classes).to(device)
        else:
            self.classifier = nn.Sequential(
                nn.Linear(feature_dim, self.hidden_dim),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(self.hidden_dim, self.num_classes),
            ).to(device)

    def build_phase_features(
        self,
        per_time_features: dict[str, torch.Tensor | None],
        scalar_features: dict[str, torch.Tensor | None],
        batch_size: int | None = None,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        """Pool and concatenate the configured phase-aware features."""
        cfg = self.config
        pieces: list[torch.Tensor] = []

        if cfg.use_hidden_magnitude and per_time_features.get("hidden_magnitude") is not None:
            pieces.extend(_pool_timeseries(per_time_features["hidden_magnitude"], cfg.pooling))
        if cfg.use_hidden_phase and per_time_features.get("hidden_phase_change") is not None:
            pieces.extend(_pool_timeseries(per_time_features["hidden_phase_change"], cfg.pooling))
        if cfg.use_gyro_magnitude and per_time_features.get("gyro_magnitude") is not None:
            pieces.extend(_pool_timeseries(per_time_features["gyro_magnitude"], cfg.pooling))
        if cfg.use_selective_score and per_time_features.get("selective_score") is not None:
            pieces.extend(_pool_timeseries(per_time_features["selective_score"], cfg.pooling))

        if cfg.use_interactions:
            p = per_time_features.get("hidden_phase_change")
            q = per_time_features.get("gyro_magnitude")
            s = per_time_features.get("selective_score")
            d = scalar_features.get("rotation_diversity")
            if cfg.use_hidden_phase and cfg.use_gyro_magnitude and p is not None and q is not None:
                # broadcast q to match p's last dim if needed
                p_red = p.mean(dim=-1) if p.dim() == 3 else p
                pq = p_red * q
                pieces.extend(_pool_timeseries(pq, cfg.pooling))
                if cfg.use_rotation_diversity and d is not None:
                    pieces.extend(_pool_timeseries(pq * d.unsqueeze(-1), cfg.pooling))
            if cfg.use_selective_score and cfg.use_gyro_magnitude and s is not None and q is not None:
                s_red = s.mean(dim=-1) if s.dim() == 3 else s
                pieces.extend(_pool_timeseries(s_red * q, cfg.pooling))
            if cfg.use_selective_score and cfg.use_hidden_phase and s is not None and p is not None:
                s_red = s.mean(dim=-1) if s.dim() == 3 else s
                p_red = p.mean(dim=-1) if p.dim() == 3 else p
                pieces.extend(_pool_timeseries(s_red * p_red, cfg.pooling))

        if cfg.use_rotation_diversity and scalar_features.get("rotation_diversity") is not None:
            pieces.append(scalar_features["rotation_diversity"].unsqueeze(-1))

        if not pieces:
            if batch_size is None or device is None:
                # Best-effort: pull batch/device from any available tensor.
                ref = next(
                    (t for t in {**per_time_features, **scalar_features}.values() if t is not None),
                    None,
                )
                if ref is None:
                    return torch.zeros(0, 0)
                return torch.zeros(ref.shape[0], 0, device=ref.device, dtype=ref.dtype)
            return torch.zeros(batch_size, 0, device=device)
        return torch.cat(pieces, dim=-1)

    def forward(
        self,
        h_base: torch.Tensor,
        per_time_features: dict[str, torch.Tensor | None],
        scalar_features: dict[str, torch.Tensor | None],
    ) -> torch.Tensor:
        phase_feat = self.build_phase_features(
            per_time_features,
            scalar_features,
            batch_size=h_base.shape[0],
            device=h_base.device,
        )
        if self.config.use_base:
            if phase_feat.numel() == 0:
                h_final = h_base
            else:
                h_final = torch.cat([h_base, phase_feat], dim=-1)
        else:
            h_final = phase_feat

        if self.classifier is None:
            self._build_classifier(h_final.shape[-1], h_final.device)
        return self.classifier(self.dropout(h_final))


# ---------------------------------------------------------------------------
# Preset configs for the head ablation table
# ---------------------------------------------------------------------------


def head_config_for(name: str) -> HeadConfig:
    """Map a short preset name to a HeadConfig instance."""
    presets = {
        "avgpool": HeadConfig(),  # base only
        "magnitude": HeadConfig(use_hidden_magnitude=True),
        "phase": HeadConfig(use_hidden_phase=True),
        "gyrophase": HeadConfig(
            use_hidden_magnitude=True,
            use_hidden_phase=True,
            use_gyro_magnitude=True,
            use_interactions=True,
        ),
        "gyrophase_rd": HeadConfig(
            use_hidden_magnitude=True,
            use_hidden_phase=True,
            use_gyro_magnitude=True,
            use_rotation_diversity=True,
            use_interactions=True,
        ),
        "gyrophase_rd_bin": HeadConfig(
            use_hidden_magnitude=True,
            use_hidden_phase=True,
            use_gyro_magnitude=True,
            use_rotation_diversity=True,
            use_interactions=True,
            rd_kind="bin",
        ),
        "selective_gyrophase": HeadConfig(
            use_hidden_magnitude=True,
            use_hidden_phase=True,
            use_gyro_magnitude=True,
            use_rotation_diversity=True,
            use_selective_score=True,
            use_interactions=True,
        ),
        # exp_plan4 §1 — Selective GyroPhase v2 / v3 with improved proxies
        "selective_gyrophase_v2": HeadConfig(
            use_hidden_magnitude=True,
            use_hidden_phase=True,
            use_gyro_magnitude=True,
            use_rotation_diversity=True,
            use_selective_score=True,
            use_interactions=True,
            selective_proxy="update_budget",
        ),
        "selective_gyrophase_v3": HeadConfig(
            use_hidden_magnitude=True,
            use_hidden_phase=True,
            use_gyro_magnitude=True,
            use_rotation_diversity=True,
            use_selective_score=True,
            use_interactions=True,
            selective_proxy="phase_velocity",
        ),
    }
    if name not in presets:
        raise ValueError(f"Unknown head preset '{name}'. Known: {sorted(presets)}.")
    return presets[name]
