"""Real-valued vs Complex-valued data-dependent SSM blocks.

Faithful to exp_plan2.md §3.3 / §3.4: the only intentional difference between
RealSSMBlock and ComplexSSMBlock is the form of the state update.

  Real:    h_t = a_t * h_{t-1} + b_t * u_t
  Complex: z_t = rho_t * exp(i theta_t) * z_{t-1} + u_t           (split real/imag)

Both blocks use data-dependent gates/projections of identical structure so any
performance gap is attributable to the update law itself. We parallelise the
linear scan over the batch dimension only — the time loop stays Pythonic, which
is fine for the short (<=256) IMU windows we use.
"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


class RealSSMBlock(nn.Module):
    """h_t = sigmoid(a) * h_{t-1} + b * u_t, with data-dependent a, b, u."""

    def __init__(self, d_model: int, d_state: int, dropout: float = 0.0,
                 a_init_bias: float = 3.0):
        super().__init__()
        self.d_state = d_state
        self.in_proj = nn.Linear(d_model, d_state)
        self.a_proj = nn.Linear(d_model, d_state)
        self.b_proj = nn.Linear(d_model, d_state)
        self.out_proj = nn.Linear(d_state, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        # Match ComplexSSMBlock's rho retention init (sigmoid(3)≈0.95) so the
        # ablation between real vs complex update is not confounded by gate
        # initialisation.
        nn.init.constant_(self.a_proj.bias, a_init_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        B, T, _ = x.shape
        a = torch.sigmoid(self.a_proj(x))            # (B, T, d_state)
        b = self.b_proj(x)                           # (B, T, d_state)
        u = self.in_proj(x)                          # (B, T, d_state)

        h = torch.zeros(B, self.d_state, device=x.device, dtype=x.dtype)
        outputs = []
        for t in range(T):
            h = a[:, t] * h + b[:, t] * u[:, t]
            outputs.append(h)
        H = torch.stack(outputs, dim=1)              # (B, T, d_state)
        y = self.out_proj(H)
        return residual + self.dropout(y)


class ComplexSSMBlock(nn.Module):
    """z_t = rho_t * exp(i theta_t) * z_{t-1} + (u_real + i u_imag).

    real/imag halves are concatenated for the output projection, so the
    parameter count is the same order as RealSSMBlock with 2x d_state.
    """

    def __init__(
        self,
        d_model: int,
        d_state: int,
        dropout: float = 0.0,
        expose_hidden: bool = False,
        rho_init_bias: float = 3.0,
        theta_init_scale: float = 0.01,
        theta_range: float = math.pi / 2,
    ):
        super().__init__()
        self.d_state = d_state
        self.theta_range = theta_range
        # input is split into real/imag halves
        self.in_proj = nn.Linear(d_model, d_state * 2)
        self.rho_proj = nn.Linear(d_model, d_state)
        self.theta_proj = nn.Linear(d_model, d_state)
        self.out_proj = nn.Linear(d_state * 2, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.expose_hidden = expose_hidden
        # rho≈0.95 at init so the hidden state retains information across the
        # full window instead of decaying with the default sigmoid(0)=0.5.
        nn.init.constant_(self.rho_proj.bias, rho_init_bias)
        # theta≈0 at init so the rotation starts close to identity and each
        # dimension can learn its own slow phase drift, mirroring the diagonal
        # complex-state init used by S4D / Mamba-style models.
        nn.init.uniform_(self.theta_proj.weight, -theta_init_scale, theta_init_scale)
        nn.init.zeros_(self.theta_proj.bias)
        # When expose_hidden=True the most recent forward stashes the per-step
        # real/imag hidden tensors so a downstream hook / analysis can read them.
        self._last_hidden_real: torch.Tensor | None = None
        self._last_hidden_imag: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        B, T, _ = x.shape
        rho = torch.sigmoid(self.rho_proj(x))                  # (B, T, d_state)
        theta = torch.tanh(self.theta_proj(x)) * self.theta_range
        u = self.in_proj(x)                                    # (B, T, 2*d_state)
        u_real, u_imag = u.chunk(2, dim=-1)

        cos_t = torch.cos(theta)
        sin_t = torch.sin(theta)

        real = torch.zeros(B, self.d_state, device=x.device, dtype=x.dtype)
        imag = torch.zeros(B, self.d_state, device=x.device, dtype=x.dtype)
        reals: list[torch.Tensor] = []
        imags: list[torch.Tensor] = []
        for t in range(T):
            r_prev, i_prev = real, imag
            real = rho[:, t] * (cos_t[:, t] * r_prev - sin_t[:, t] * i_prev) + u_real[:, t]
            imag = rho[:, t] * (sin_t[:, t] * r_prev + cos_t[:, t] * i_prev) + u_imag[:, t]
            reals.append(real)
            imags.append(imag)
        H_real = torch.stack(reals, dim=1)                     # (B, T, d_state)
        H_imag = torch.stack(imags, dim=1)
        if self.expose_hidden:
            self._last_hidden_real = H_real.detach()
            self._last_hidden_imag = H_imag.detach()
        h = torch.cat([H_real, H_imag], dim=-1)
        y = self.out_proj(h)
        return residual + self.dropout(y)


class _SSMClassifier(nn.Module):
    """Common encoder for the Real / Complex SSM ablation classifiers."""

    def __init__(
        self,
        block_cls: type[nn.Module],
        in_channels: int,
        num_classes: int,
        d_model: int,
        d_state: int,
        n_layers: int,
        dropout: float,
        block_kwargs: dict | None = None,
    ):
        super().__init__()
        self.input_proj = nn.Linear(in_channels, d_model)
        block_kwargs = block_kwargs or {}
        self.blocks = nn.ModuleList(
            [block_cls(d_model=d_model, d_state=d_state, dropout=dropout, **block_kwargs) for _ in range(n_layers)]
        )
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(d_model, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_proj(x.float())
        for block in self.blocks:
            h = block(h)
        h = self.norm(h)
        pooled = h[:, -1, :]
        return self.head(self.dropout(pooled))


class RealSSMClassifier(_SSMClassifier):
    def __init__(self, in_channels: int, num_classes: int = 2, d_model: int = 64,
                 d_state: int = 64, n_layers: int = 2, dropout: float = 0.1):
        super().__init__(
            block_cls=RealSSMBlock,
            in_channels=in_channels,
            num_classes=num_classes,
            d_model=d_model,
            d_state=d_state,
            n_layers=n_layers,
            dropout=dropout,
        )


class ComplexSSMClassifier(_SSMClassifier):
    def __init__(self, in_channels: int, num_classes: int = 2, d_model: int = 64,
                 d_state: int = 32, n_layers: int = 2, dropout: float = 0.1,
                 expose_hidden: bool = False):
        # d_state for ComplexSSM is half of RealSSM's, because real/imag concat
        # gives a 2*d_state-wide hidden tensor before out_proj. This keeps the
        # output-projection parameter count roughly aligned.
        super().__init__(
            block_cls=ComplexSSMBlock,
            in_channels=in_channels,
            num_classes=num_classes,
            d_model=d_model,
            d_state=d_state,
            n_layers=n_layers,
            dropout=dropout,
            block_kwargs={"expose_hidden": expose_hidden},
        )
