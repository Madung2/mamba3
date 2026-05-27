"""End-to-end classifier: encoder backbone + (optional) GyroPhase Head.

The classifier handles all the bookkeeping needed to feed phase-aware
features into the head:

  - normalises raw IMU input (used as the encoder feed),
  - keeps a *denormalised* copy of the input so we can compute gyro magnitude
    and rotation diversity in physical units,
  - extracts hidden magnitude / phase change from the encoder's state dict
    when the backbone exposes complex hidden state,
  - delegates the actual feature selection to ``GyroPhaseHead``.

Models that don't expose complex state (Transformer / TCN / Mamba-3 / Real-*)
still benefit from gyro magnitude + rotation diversity, but their hidden-phase
features fall back to zeros.
"""

from __future__ import annotations

import torch
from torch import nn

from .encoders import build_backbone
from .gyrophase import (
    GyroPhaseHead,
    HeadConfig,
    compute_gyro_magnitude,
    compute_phase_change,
    compute_rotation_diversity_bin,
    compute_rotation_diversity_std,
)


class PhaseAwareClassifier(nn.Module):
    """Glue module: backbone(x) → pooled feature + phase head."""

    def __init__(
        self,
        backbone_name: str,
        head_config: HeadConfig,
        in_channels: int,
        num_classes: int,
        gyro_indices: tuple[int, ...] | None,
        backbone_config: dict | None = None,
        dropout: float = 0.1,
        pool: str = "mean",
    ):
        super().__init__()
        self.backbone_name = backbone_name
        self.head_config = head_config
        self.gyro_indices = gyro_indices
        self.pool = pool

        self.backbone = build_backbone(backbone_name, in_channels=in_channels,
                                       model_config=backbone_config)
        self.head = GyroPhaseHead(
            d_base=self.backbone.d_base,
            num_classes=num_classes,
            config=head_config,
            dropout=dropout,
        )

    @staticmethod
    def _pool_sequence(h: torch.Tensor, mode: str) -> torch.Tensor:
        if mode == "mean":
            return h.mean(dim=1)
        if mode == "last":
            return h[:, -1, :]
        if mode == "max":
            return h.max(dim=1).values
        raise ValueError(f"Unknown pool mode '{mode}'.")

    def _build_per_time_features(
        self, x_raw: torch.Tensor, state: dict[str, torch.Tensor]
    ) -> tuple[dict[str, torch.Tensor | None], dict[str, torch.Tensor | None]]:
        cfg = self.head_config
        T = x_raw.shape[1]
        per_time: dict[str, torch.Tensor | None] = {
            "hidden_magnitude": None,
            "hidden_phase_change": None,
            "gyro_magnitude": None,
            "selective_score": None,
        }
        scalar: dict[str, torch.Tensor | None] = {"rotation_diversity": None}

        h_real = state.get("h_real")
        h_imag = state.get("h_imag")
        sel = state.get("selective_score")

        if cfg.use_hidden_magnitude and h_real is not None and h_imag is not None:
            mag = (h_real ** 2 + h_imag ** 2).sqrt()                # (B, T, d_state)
            per_time["hidden_magnitude"] = mag.mean(dim=-1)         # (B, T)
        elif cfg.use_hidden_magnitude and h_real is not None:
            per_time["hidden_magnitude"] = h_real.abs().mean(dim=-1)

        if cfg.use_hidden_phase:
            if h_real is not None and h_imag is not None:
                pc = compute_phase_change(h_real, h_imag).mean(dim=-1)  # (B, T-1)
                pad = torch.zeros(pc.shape[0], 1, device=pc.device, dtype=pc.dtype)
                per_time["hidden_phase_change"] = torch.cat([pad, pc], dim=1)
            else:
                # Backbone does not expose complex state — fall back to
                # gyro-derived phase. This is the explicit "Mamba-3 + GyroPhase"
                # routing flagged in the project memo.
                per_time["hidden_phase_change"] = self._fallback_gyro_phase_change(x_raw)

        if cfg.use_gyro_magnitude or cfg.use_interactions:
            per_time["gyro_magnitude"] = compute_gyro_magnitude(x_raw, self.gyro_indices)

        if cfg.use_selective_score and sel is not None:
            per_time["selective_score"] = sel.mean(dim=-1)

        if cfg.use_rotation_diversity or cfg.use_interactions:
            if cfg.rd_kind == "bin":
                rd = compute_rotation_diversity_bin(x_raw, self.gyro_indices)
            else:
                rd = compute_rotation_diversity_std(x_raw, self.gyro_indices)
            scalar["rotation_diversity"] = rd

        return per_time, scalar

    @staticmethod
    def _fallback_gyro_phase_change(x_raw: torch.Tensor) -> torch.Tensor:
        """Approximate per-step rotation magnitude from raw gyro input.

        For an encoder that does not expose hidden phase we use the magnitude
        of the windowed gyro vector change as a phase-change proxy. This is
        documented as the "no internal state" routing for Mamba-3 etc.
        """
        if x_raw.shape[-1] < 6:
            return torch.zeros(x_raw.shape[0], x_raw.shape[1], device=x_raw.device, dtype=x_raw.dtype)
        gyro = x_raw[..., 3:6]
        diff = torch.zeros_like(gyro[..., :1])
        diff[:, 1:, 0] = (gyro[:, 1:] - gyro[:, :-1]).norm(dim=-1)
        return diff.squeeze(-1)

    def forward(self, x_raw: torch.Tensor) -> torch.Tensor:
        """Returns logits."""
        h_seq, state = self.backbone(x_raw)
        h_base = self._pool_sequence(h_seq, self.pool)
        per_time, scalar = self._build_per_time_features(x_raw, state)
        return self.head(h_base, per_time, scalar)

    def forward_with_state(
        self, x_raw: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Variant that also returns the raw state dict for analysis."""
        h_seq, state = self.backbone(x_raw)
        h_base = self._pool_sequence(h_seq, self.pool)
        per_time, scalar = self._build_per_time_features(x_raw, state)
        logits = self.head(h_base, per_time, scalar)
        return logits, {"per_time": per_time, "scalar": scalar, "state": state, "h_base": h_base}
