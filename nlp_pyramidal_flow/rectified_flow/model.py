from __future__ import annotations

import math
from dataclasses import dataclass


def _require_torch():
    try:
        import torch  # noqa: F401
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "缺少 torch 依赖：请先 `pip install torch`（或按 requirements.txt 安装）。"
        ) from e


@dataclass
class RFModelConfig:
    dim: int
    hidden: int = 256
    depth: int = 3
    dropout: float = 0.0


def sinusoidal_t_embedding(t, dim: int):
    """
    Standard sinusoidal embedding for scalar t in [0,1].
    Returns shape [B, dim].
    """
    _require_torch()
    import torch

    half = dim // 2
    freqs = torch.exp(
        torch.linspace(0, math.log(10_000), half, device=t.device)
    )  # [half]
    # scale t into radians
    args = t[:, None] * freqs[None, :]  # [B, half]
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if dim % 2 == 1:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb


def build_rf_mlp(cfg: RFModelConfig):
    """
    v_theta(x_t, t) -> u_t
    where Rectified Flow target is u_t = x1 - x0 (constant in t).
    """
    _require_torch()
    import torch
    import torch.nn as nn

    t_dim = min(128, max(16, cfg.hidden // 2))
    layers = []
    in_dim = cfg.dim + t_dim
    for i in range(cfg.depth):
        out_dim = cfg.hidden if i < cfg.depth - 1 else cfg.dim
        layers.append(nn.Linear(in_dim, out_dim))
        if i < cfg.depth - 1:
            layers.append(nn.SiLU())
            if cfg.dropout and cfg.dropout > 0:
                layers.append(nn.Dropout(cfg.dropout))
        in_dim = cfg.hidden
    net = nn.Sequential(*layers)

    class RFWrapper(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = net
            self.t_dim = t_dim

        def forward(self, x, t):
            # x: [B, D], t: [B]
            te = sinusoidal_t_embedding(t, self.t_dim)
            inp = torch.cat([x, te], dim=-1)
            return self.net(inp)

    return RFWrapper()


