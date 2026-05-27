"""Backbone encoders that return (sequence features, internal state dict).

Each encoder produces:
  - ``h_seq``: (B, T, d_base) sequence features after final layer norm.
  - ``state``: optional dict with ``h_real``, ``h_imag``, ``selective_score`` —
    only populated by SSM blocks that track them.

Heads operate on ``mean(h_seq)`` plus phase-aware features derived from the
state dict and the raw input. This keeps the boundary clean and lets us swap
backbones (TCN, Transformer, Mamba-3, 2x2 SSMs) under the same training loop.
"""

from __future__ import annotations

import torch
from torch import nn

from .baselines import TemporalBlock
from .ssm_2x2 import SSM2x2Encoder
from .ssm_ablation import ComplexSSMBlock, RealSSMBlock


class _SequenceModel(nn.Module):
    """Marker base class so the trainer can spot encoder-style modules."""

    d_base: int

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        raise NotImplementedError


class TransformerEncoderBackbone(_SequenceModel):
    def __init__(
        self,
        in_channels: int,
        d_model: int = 64,
        num_heads: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
        max_len: int = 256,
    ):
        super().__init__()
        self.d_base = d_model
        self.input_proj = nn.Linear(in_channels, d_model)
        self.pos_emb = nn.Parameter(torch.zeros(1, max_len, d_model))
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        nn.init.normal_(self.pos_emb, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        seq_len = x.shape[1]
        x = self.input_proj(x) + self.pos_emb[:, :seq_len, :]
        h = self.encoder(x)
        h = self.norm(h)
        return h, {}


class TCNBackbone(_SequenceModel):
    def __init__(
        self,
        in_channels: int,
        channels: tuple[int, ...] = (64, 64, 128, 128),
        kernel_size: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        blocks = []
        cur = in_channels
        for level, out_c in enumerate(channels):
            blocks.append(
                TemporalBlock(
                    in_channels=cur,
                    out_channels=out_c,
                    kernel_size=kernel_size,
                    dilation=2 ** level,
                    dropout=dropout,
                )
            )
            cur = out_c
        self.encoder = nn.Sequential(*blocks)
        self.d_base = cur

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        h = self.encoder(x.transpose(1, 2))
        h = h.transpose(1, 2)  # (B, T, C)
        return h, {}


class Mamba3Backbone(_SequenceModel):
    """Wraps the official Mamba-3 mixer behind the encoder interface."""

    def __init__(
        self,
        in_channels: int,
        d_model: int = 128,
        d_state: int = 64,
        expand: int = 2,
        headdim: int = 64,
        n_layers: int = 2,
        dropout: float = 0.1,
        chunk_size: int = 16,
    ):
        super().__init__()
        from .mamba3_classifier import Mamba3ResidualBlock  # avoid circular at import

        self.d_base = d_model
        self.input_proj = nn.Linear(in_channels, d_model)
        self.blocks = nn.ModuleList(
            [
                Mamba3ResidualBlock(
                    d_model=d_model,
                    d_state=d_state,
                    expand=expand,
                    headdim=headdim,
                    chunk_size=chunk_size,
                    dropout=dropout,
                )
                for _ in range(n_layers)
            ]
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if not x.is_cuda:
            raise RuntimeError("Mamba-3 backbone requires CUDA for the Triton kernel.")
        h = self.input_proj(x.float())
        for block in self.blocks:
            h = block(h)
        return self.norm(h), {}


class _GenericSSMBackbone(_SequenceModel):
    """Wraps the original Real/Complex SSM blocks under the encoder interface."""

    def __init__(self, block_cls: type[nn.Module], in_channels: int,
                 d_model: int = 64, d_state: int = 64, n_layers: int = 2,
                 dropout: float = 0.1, expose_hidden: bool = False):
        super().__init__()
        self.d_base = d_model
        self.input_proj = nn.Linear(in_channels, d_model)
        block_kwargs = {"expose_hidden": expose_hidden} if block_cls is ComplexSSMBlock else {}
        self.blocks = nn.ModuleList(
            [
                block_cls(d_model=d_model, d_state=d_state, dropout=dropout, **block_kwargs)
                for _ in range(n_layers)
            ]
        )
        self.norm = nn.LayerNorm(d_model)
        self._is_complex = block_cls is ComplexSSMBlock
        if self._is_complex:
            # ensure the final block exposes its hidden state
            self.blocks[-1].expose_hidden = True

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        h = self.input_proj(x.float())
        for block in self.blocks:
            h = block(h)
        h = self.norm(h)
        state: dict[str, torch.Tensor] = {}
        if self._is_complex:
            last = self.blocks[-1]
            if last._last_hidden_real is not None:
                state["h_real"] = last._last_hidden_real
                state["h_imag"] = last._last_hidden_imag
        return h, state


class RealSSMBackbone(_GenericSSMBackbone):
    def __init__(self, in_channels: int, d_model: int = 64, d_state: int = 64,
                 n_layers: int = 2, dropout: float = 0.1):
        super().__init__(RealSSMBlock, in_channels, d_model, d_state, n_layers, dropout)


class ComplexSSMBackbone(_GenericSSMBackbone):
    def __init__(self, in_channels: int, d_model: int = 64, d_state: int = 64,
                 n_layers: int = 2, dropout: float = 0.1):
        super().__init__(ComplexSSMBlock, in_channels, d_model, d_state, n_layers, dropout,
                         expose_hidden=True)


class Ablation2x2Backbone(_SequenceModel):
    """Backbone wrapping a single 2x2 ablation choice."""

    def __init__(self, block_name: str, in_channels: int, d_model: int = 64,
                 d_state: int = 64, n_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.encoder = SSM2x2Encoder(
            block_name=block_name,
            in_channels=in_channels,
            d_model=d_model,
            d_state=d_state,
            n_layers=n_layers,
            dropout=dropout,
        )
        self.d_base = d_model

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        return self.encoder(x)


def build_backbone(name: str, in_channels: int, model_config: dict | None = None) -> _SequenceModel:
    model_config = model_config or {}
    if name == "transformer":
        return TransformerEncoderBackbone(in_channels=in_channels, **model_config)
    if name == "tcn":
        return TCNBackbone(in_channels=in_channels, **model_config)
    if name == "mamba3":
        return Mamba3Backbone(in_channels=in_channels, **model_config)
    if name == "real_ssm":
        return RealSSMBackbone(in_channels=in_channels, **model_config)
    if name == "complex_ssm":
        return ComplexSSMBackbone(in_channels=in_channels, **model_config)
    if name in {"real_static", "real_selective", "complex_static", "complex_selective"}:
        return Ablation2x2Backbone(block_name=name, in_channels=in_channels, **model_config)
    raise ValueError(f"Unknown backbone '{name}'.")
