from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np


def _require_torch():
    try:
        import torch  # noqa: F401
    except Exception as e:
        raise RuntimeError("缺少 torch 依赖，请先安装 requirements.txt。") from e


def _try_import_torchdyn():
    try:
        from torchdyn.core import NeuralODE

        return NeuralODE
    except Exception:
        return None


@dataclass
class ODESolveConfig:
    steps: int = 20
    method: str = "rk4"  # if torchdyn present


def solve_trajectory(
    x_start: np.ndarray,
    model: Any,
    device: str = "cpu",
    cfg: Optional[ODESolveConfig] = None,
    direction: str = "forward",
) -> np.ndarray:
    """
    Return the full trajectory embeddings.

    - forward: integrate t:0->1, dx/dt = v_theta(x,t)
    - reverse: integrate t:0->1, dx/dt = -v_theta(x, 1-t)
      (a simple time-reversal for demonstration; helps visualize "clean -> noisy" too)

    Output shape: [steps+1, B, D]
    """
    _require_torch()
    import torch

    if cfg is None:
        cfg = ODESolveConfig()

    x0 = torch.from_numpy(x_start.astype(np.float32)).to(device)
    NeuralODE = _try_import_torchdyn()

    def vf_eval(t_scalar, x):
        if direction == "forward":
            tt = torch.ones(x.shape[0], device=x.device) * t_scalar
            return model(x, tt)
        # reverse
        tt = torch.ones(x.shape[0], device=x.device) * (1.0 - t_scalar)
        return -model(x, tt)

    if NeuralODE is not None:
        class VF(torch.nn.Module):
            def __init__(self, net):
                super().__init__()
                self.net = net

            def forward(self, t, x, args=None, **kwargs):
                return vf_eval(float(t), x)

        ode = NeuralODE(VF(model), solver=cfg.method, sensitivity="autograd", atol=1e-5, rtol=1e-5)
        t_span = torch.linspace(0, 1, int(cfg.steps) + 1, device=x0.device)
        traj = ode.trajectory(x0, t_span)  # [T, B, D]
        return traj.detach().cpu().numpy()

    # fallback Euler with logging
    steps = int(cfg.steps)
    dt = 1.0 / max(1, steps)
    xs = [x0]
    t = 0.0
    x = x0
    for _ in range(steps):
        dx = vf_eval(t, x)
        x = x + dt * dx
        xs.append(x)
        t += dt
    traj = torch.stack(xs, dim=0)  # [T, B, D]
    return traj.detach().cpu().numpy()


def solve_to_clean_embedding(
    x0: np.ndarray,
    model: Any,
    device: str = "cpu",
    cfg: Optional[ODESolveConfig] = None,
) -> np.ndarray:
    """
    Integrate dx/dt = v_theta(x,t), t:0->1, start from x0 (noisy embedding).
    Output x1_hat (clean embedding estimate).

    Uses torchdyn if available; otherwise uses a simple Euler solver.
    """
    _require_torch()
    import torch

    if cfg is None:
        cfg = ODESolveConfig()

    x = torch.from_numpy(x0.astype(np.float32)).to(device)

    NeuralODE = _try_import_torchdyn()
    if NeuralODE is not None:
        # torchdyn expects a vector field f(t, x)
        class VF(torch.nn.Module):
            def __init__(self, net):
                super().__init__()
                self.net = net

            def forward(self, t, x, args=None, **kwargs):
                # t: scalar tensor, x: [B, D]
                # torchdyn may pass extra kwargs (e.g., args=...) depending on version.
                # We don't use them for this unconditional vector field.
                tt = torch.ones(x.shape[0], device=x.device) * t
                return self.net(x, tt)

        vf = VF(model)
        ode = NeuralODE(vf, solver=cfg.method, sensitivity="autograd", atol=1e-5, rtol=1e-5)
        t_span = torch.linspace(0, 1, int(cfg.steps) + 1, device=x.device)
        traj = ode.trajectory(x, t_span)
        x1 = traj[-1]
        return x1.detach().cpu().numpy()

    # fallback: Euler
    steps = int(cfg.steps)
    dt = 1.0 / max(1, steps)
    t = 0.0
    for _ in range(steps):
        tt = torch.ones(x.shape[0], device=x.device) * t
        dx = model(x, tt)
        x = x + dt * dx
        t += dt
    return x.detach().cpu().numpy()


