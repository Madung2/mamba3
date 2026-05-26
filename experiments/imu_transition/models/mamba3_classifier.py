"""Mamba-3 classifier for IMU transition detection."""

from __future__ import annotations

import torch
from torch import nn

from mamba_ssm.modules.mamba3 import Mamba3


class Mamba3ResidualBlock(nn.Module):
    """A lightweight residual wrapper around a Mamba-3 mixer."""

    def __init__(
        self,
        d_model: int,
        d_state: int,
        expand: int,
        headdim: int,
        chunk_size: int,
        dropout: float,
    ):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.mixer = Mamba3(
            d_model=d_model,
            d_state=d_state,
            expand=expand,
            headdim=headdim,
            chunk_size=chunk_size,
            is_mimo=False,
            dtype=torch.bfloat16,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        h = self.norm(x).to(dtype=torch.bfloat16)
        h = self.mixer(h).float()
        return residual + self.dropout(h)


class Mamba3TransitionClassifier(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_classes: int = 2,
        d_model: int = 128,
        d_state: int = 64,
        expand: int = 2,
        headdim: int = 64,
        n_layers: int = 2,
        dropout: float = 0.1,
        chunk_size: int = 16,
    ):
        super().__init__()
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
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(d_model, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not x.is_cuda:
            raise RuntimeError("Mamba3TransitionClassifier requires a CUDA device for the Triton kernel.")

        h = self.input_proj(x.float())
        for block in self.blocks:
            h = block(h)
        h = self.norm(h)
        pooled = h[:, -1, :]
        return self.head(self.dropout(pooled))
