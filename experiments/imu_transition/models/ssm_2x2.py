"""2x2 SSM ablation: {Real, Complex} x {Static, Selective}.

Goal: separate the effect of *complex* state update (rotation) from the effect
of *selective* (input-dependent) update strength. Each block exposes per-step
hidden state and a `selective_score` proxy so the GyroPhase Head can ingest
them.

Static blocks share the *parameter shape* of their selective counterparts but
the `a` / `rho` / `theta` factors are computed from a learned channel-wise
parameter rather than from a data-dependent projection. This isolates the
selective-scan effect.
"""

from __future__ import annotations

import math

import torch
from torch import nn


def _make_input_projection(in_dim: int, out_dim: int) -> nn.Linear:
    return nn.Linear(in_dim, out_dim)


class _RealSSMBase(nn.Module):
    """Shared scaffolding for real-valued ablation blocks."""

    def __init__(
        self,
        d_model: int,
        d_state: int,
        dropout: float,
        selective: bool,
        a_init_bias: float = 3.0,
    ):
        super().__init__()
        self.d_state = d_state
        self.selective = selective
        self.norm = nn.LayerNorm(d_model)
        self.in_proj = nn.Linear(d_model, d_state)
        self.b_proj = nn.Linear(d_model, d_state)
        self.out_proj = nn.Linear(d_state, d_model)
        self.dropout = nn.Dropout(dropout)
        if selective:
            self.a_proj = nn.Linear(d_model, d_state)
            nn.init.constant_(self.a_proj.bias, a_init_bias)
        else:
            # channel-wise static logit
            self.a_logit = nn.Parameter(torch.full((d_state,), a_init_bias))
        # buffers populated when expose_hidden=True
        self.expose_hidden = False
        self._last_state: dict[str, torch.Tensor] = {}

    def _gate(self, x: torch.Tensor) -> torch.Tensor:
        if self.selective:
            return torch.sigmoid(self.a_proj(x))
        return torch.sigmoid(self.a_logit).expand(x.size(0), x.size(1), -1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        B, T, _ = x.shape
        a = self._gate(x)                              # (B, T, d_state)
        b = self.b_proj(x)
        u = self.in_proj(x)

        h = torch.zeros(B, self.d_state, device=x.device, dtype=x.dtype)
        outputs = []
        for t in range(T):
            h = a[:, t] * h + b[:, t] * u[:, t]
            outputs.append(h)
        H = torch.stack(outputs, dim=1)                # (B, T, d_state)
        if self.expose_hidden:
            update_strength = (b * u).abs().mean(dim=-1, keepdim=False)   # (B, T)
            self._last_state = {
                "h_real": H.detach(),
                "h_imag": torch.zeros_like(H).detach(),
                "selective_score": a.detach(),
                "update_strength": update_strength.detach(),
                "retention": a.detach(),
            }
        y = self.out_proj(H)
        return residual + self.dropout(y)


class RealStaticSSMBlock(_RealSSMBase):
    def __init__(self, d_model: int, d_state: int, dropout: float = 0.0):
        super().__init__(d_model=d_model, d_state=d_state, dropout=dropout, selective=False)


class RealSelectiveSSMBlock(_RealSSMBase):
    def __init__(self, d_model: int, d_state: int, dropout: float = 0.0):
        super().__init__(d_model=d_model, d_state=d_state, dropout=dropout, selective=True)


class _ComplexSSMBase(nn.Module):
    """Shared complex-state scaffolding.

    Complex state update:
        z_t = rho_t * exp(i theta_t) * z_{t-1} + u_t

    Static variant: rho, theta come from channel-wise parameters.
    Selective variant: rho, theta come from data-dependent projections.
    """

    def __init__(
        self,
        d_model: int,
        d_state: int,
        dropout: float,
        selective: bool,
        rho_init_bias: float = 3.0,
        theta_init_scale: float = 0.01,
        theta_range: float = math.pi / 2,
    ):
        super().__init__()
        self.d_state = d_state
        self.theta_range = theta_range
        self.selective = selective
        self.norm = nn.LayerNorm(d_model)
        self.in_proj = nn.Linear(d_model, d_state * 2)  # real/imag input
        self.out_proj = nn.Linear(d_state * 2, d_model)
        self.dropout = nn.Dropout(dropout)
        if selective:
            self.rho_proj = nn.Linear(d_model, d_state)
            self.theta_proj = nn.Linear(d_model, d_state)
            nn.init.constant_(self.rho_proj.bias, rho_init_bias)
            nn.init.uniform_(self.theta_proj.weight, -theta_init_scale, theta_init_scale)
            nn.init.zeros_(self.theta_proj.bias)
        else:
            self.rho_logit = nn.Parameter(torch.full((d_state,), rho_init_bias))
            # Static theta initialised to small random per-channel value so the
            # rotation isn't degenerate at init.
            theta = (torch.rand(d_state) * 2 - 1) * theta_init_scale
            self.theta_param = nn.Parameter(theta)
        self.expose_hidden = False
        self._last_state: dict[str, torch.Tensor] = {}

    def _rotation(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.selective:
            rho = torch.sigmoid(self.rho_proj(x))
            theta = torch.tanh(self.theta_proj(x)) * self.theta_range
        else:
            rho = torch.sigmoid(self.rho_logit).expand(x.size(0), x.size(1), -1)
            theta = (torch.tanh(self.theta_param) * self.theta_range).expand(
                x.size(0), x.size(1), -1
            )
        return rho, theta

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        B, T, _ = x.shape
        rho, theta = self._rotation(x)
        u = self.in_proj(x)
        u_real, u_imag = u.chunk(2, dim=-1)
        cos_t = torch.cos(theta)
        sin_t = torch.sin(theta)

        real = torch.zeros(B, self.d_state, device=x.device, dtype=x.dtype)
        imag = torch.zeros(B, self.d_state, device=x.device, dtype=x.dtype)
        reals, imags = [], []
        for t in range(T):
            r_prev, i_prev = real, imag
            real = rho[:, t] * (cos_t[:, t] * r_prev - sin_t[:, t] * i_prev) + u_real[:, t]
            imag = rho[:, t] * (sin_t[:, t] * r_prev + cos_t[:, t] * i_prev) + u_imag[:, t]
            reals.append(real)
            imags.append(imag)
        H_real = torch.stack(reals, dim=1)
        H_imag = torch.stack(imags, dim=1)
        if self.expose_hidden:
            mag = (u_real ** 2 + u_imag ** 2).sqrt()                 # (B, T, d_state)
            forget = 1.0 - rho
            update_budget = forget * mag
            # phase velocity ≈ rho * |sin(theta)| as a proxy for the actual
            # per-step rotation angle applied to the previous state.
            phase_velocity = rho * torch.abs(torch.sin(theta))
            self._last_state = {
                "h_real": H_real.detach(),
                "h_imag": H_imag.detach(),
                "selective_score": rho.detach(),         # legacy proxy
                "update_strength": mag.detach(),
                "retention": rho.detach(),
                "phase_step": theta.detach(),
                # exp_plan4 §1 — new proxies
                "forget_rate": forget.detach(),
                "update_budget": update_budget.detach(),
                "phase_velocity": phase_velocity.detach(),
            }
        h = torch.cat([H_real, H_imag], dim=-1)
        y = self.out_proj(h)
        return residual + self.dropout(y)


class ComplexStaticSSMBlock(_ComplexSSMBase):
    def __init__(self, d_model: int, d_state: int, dropout: float = 0.0):
        super().__init__(d_model=d_model, d_state=d_state, dropout=dropout, selective=False)


class ComplexSelectiveSSMBlock(_ComplexSSMBase):
    def __init__(self, d_model: int, d_state: int, dropout: float = 0.0):
        super().__init__(d_model=d_model, d_state=d_state, dropout=dropout, selective=True)


_BLOCK_REGISTRY = {
    "real_static": RealStaticSSMBlock,
    "real_selective": RealSelectiveSSMBlock,
    "complex_static": ComplexStaticSSMBlock,
    "complex_selective": ComplexSelectiveSSMBlock,
}


def make_block(name: str, d_model: int, d_state: int, dropout: float) -> nn.Module:
    if name not in _BLOCK_REGISTRY:
        raise ValueError(f"Unknown 2x2 SSM block '{name}'.")
    return _BLOCK_REGISTRY[name](d_model=d_model, d_state=d_state, dropout=dropout)


class SSM2x2Encoder(nn.Module):
    """Stacked 2x2 ablation blocks with optional state exposure.

    The encoder returns (per-timestep features, dict of last-block states).
    The last block always runs with ``expose_hidden=True`` so downstream heads
    can read its hidden state without re-running the scan.
    """

    def __init__(
        self,
        block_name: str,
        in_channels: int,
        d_model: int = 64,
        d_state: int = 64,
        n_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.block_name = block_name
        self.input_proj = nn.Linear(in_channels, d_model)
        self.blocks = nn.ModuleList([
            make_block(block_name, d_model=d_model, d_state=d_state, dropout=dropout)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        # Expose state only on the last block.
        self.blocks[-1].expose_hidden = True

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        h = self.input_proj(x.float())
        for block in self.blocks:
            h = block(h)
        h = self.norm(h)
        return h, dict(self.blocks[-1]._last_state)
