"""Baseline models for IMU transition detection."""

from __future__ import annotations

import torch
from torch import nn


class CNN1DClassifier(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_classes: int = 2,
        hidden_channels: list[int] | tuple[int, ...] = (64, 128, 128),
        kernel_sizes: list[int] | tuple[int, ...] = (5, 3, 3),
        dropout: float = 0.1,
    ):
        super().__init__()
        if len(hidden_channels) != len(kernel_sizes):
            raise ValueError("hidden_channels and kernel_sizes must have the same length.")

        layers: list[nn.Module] = []
        current_channels = in_channels
        for out_channels, kernel_size in zip(hidden_channels, kernel_sizes):
            padding = kernel_size // 2
            layers.extend(
                [
                    nn.Conv1d(current_channels, out_channels, kernel_size=kernel_size, padding=padding),
                    nn.BatchNorm1d(out_channels),
                    nn.ReLU(inplace=True),
                    nn.Dropout(dropout),
                ]
            )
            current_channels = out_channels

        self.encoder = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(current_channels, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        x = self.encoder(x)
        x = self.pool(x).squeeze(-1)
        return self.head(x)


class GRUClassifier(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_classes: int = 2,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.gru = nn.GRU(
            input_size=in_channels,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output, _ = self.gru(x)
        pooled = self.norm(output[:, -1, :])
        return self.head(self.dropout(pooled))


class TemporalBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int, dropout: float):
        super().__init__()
        padding = dilation * (kernel_size - 1) // 2
        self.net = nn.Sequential(
            nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=padding,
                dilation=dilation,
            ),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv1d(
                out_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=padding,
                dilation=dilation,
            ),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.residual = (
            nn.Identity() if in_channels == out_channels else nn.Conv1d(in_channels, out_channels, kernel_size=1)
        )
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(self.net(x) + self.residual(x))


class TCNClassifier(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_classes: int = 2,
        channels: list[int] | tuple[int, ...] = (64, 64, 128, 128),
        kernel_size: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        blocks: list[nn.Module] = []
        current_channels = in_channels
        for level, out_channels in enumerate(channels):
            blocks.append(
                TemporalBlock(
                    in_channels=current_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    dilation=2**level,
                    dropout=dropout,
                )
            )
            current_channels = out_channels

        self.encoder = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(current_channels, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x.transpose(1, 2))
        x = self.pool(x).squeeze(-1)
        return self.head(x)


class TransformerEncoderClassifier(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_classes: int = 2,
        d_model: int = 64,
        num_heads: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
        max_len: int = 256,
    ):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.input_proj = nn.Linear(in_channels, d_model)
        self.pos_emb = nn.Parameter(torch.zeros(1, max_len, d_model))
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(d_model, num_classes)
        nn.init.normal_(self.pos_emb, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.shape[1]
        if seq_len > self.pos_emb.shape[1]:
            raise ValueError(f"Sequence length {seq_len} exceeds maximum length {self.pos_emb.shape[1]}.")
        x = self.input_proj(x) + self.pos_emb[:, :seq_len, :]
        x = self.encoder(x)
        x = self.norm(x.mean(dim=1))
        return self.head(self.dropout(x))
