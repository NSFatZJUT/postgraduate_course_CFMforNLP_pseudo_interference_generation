from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from nlp_pyramidal_flow.pyramidal_flow import PyramidalFlowPerturber
from nlp_pyramidal_flow.rectified_flow.model import RFModelConfig, build_rf_mlp
from nlp_pyramidal_flow.tasks.embeddings import TfidfEmbedder


def _try_import_torchcfm():
    try:
        from torchcfm.conditional_flow_matching import ConditionalFlowMatcher

        return ConditionalFlowMatcher
    except Exception:
        return None


def _require_torch():
    try:
        import torch  # noqa: F401
    except Exception as e:
        raise RuntimeError("缺少 torch 依赖，请先安装 requirements.txt。") from e


@dataclass
class RFTrainConfig:
    epochs: int = 10
    batch_size: int = 64
    lr: float = 1e-3
    weight_decay: float = 0.0
    device: str = "cpu"
    seed: int = 42


@dataclass
class RFArtifacts:
    embedder: TfidfEmbedder
    model: Any
    train_loss: list[float]
    meta: dict


def build_rf_dataset(
    texts: list[str],
    perturber: PyramidalFlowPerturber,
    style: str = "通用",
    noisy_per_text: int = 2,
    seed: int = 42,
) -> tuple[list[str], list[str]]:
    """
    Build (noisy_text, clean_text) pairs for RF training in embedding space:
    x0 = embed(noisy), x1 = embed(clean)
    """
    rng = np.random.default_rng(seed)
    clean = []
    noisy = []
    for t in texts:
        t = str(t).strip()
        if not t:
            continue
        for _ in range(int(noisy_per_text)):
            # randomly jitter cfg intensities a bit to increase diversity
            # (keep the same max_ops; only vary intensities)
            _ = rng.random()  # advance RNG for determinism
            out = perturber.perturb(t, style=style).text
            clean.append(t)
            noisy.append(out)
    return noisy, clean


def train_rectified_flow(
    clean_texts: list[str],
    perturber: PyramidalFlowPerturber,
    style: str,
    rf_cfg: RFModelConfig,
    train_cfg: RFTrainConfig,
    noisy_per_text: int = 2,
) -> RFArtifacts:
    """
    Rectified Flow / Flow Matching training:

    Given x0 (noisy embedding) and x1 (clean embedding), for t~U(0,1):
      x_t = (1-t)*x0 + t*x1
      u_t = x1 - x0
    Train v_theta(x_t, t) to regress u_t (MSE).

    If torchcfm is installed, we also instantiate ConditionalFlowMatcher to make the
    "flow matching" dependency explicit (but we still compute u_t as above).
    """
    _require_torch()
    import torch
    import torch.nn.functional as F

    torch.manual_seed(train_cfg.seed)
    np.random.seed(train_cfg.seed)

    # Build pairs
    noisy_texts, clean_texts2 = build_rf_dataset(
        clean_texts, perturber=perturber, style=style, noisy_per_text=noisy_per_text, seed=train_cfg.seed
    )

    # Embed
    embedder = TfidfEmbedder.fit(clean_texts2, lang_hint=perturber.cfg.lang_hint)
    x0 = embedder.transform(noisy_texts)  # noisy
    x1 = embedder.transform(clean_texts2)  # clean

    D = x0.shape[1]
    if rf_cfg.dim != D:
        rf_cfg = RFModelConfig(dim=D, hidden=rf_cfg.hidden, depth=rf_cfg.depth, dropout=rf_cfg.dropout)

    model = build_rf_mlp(rf_cfg)
    device = torch.device(train_cfg.device)
    model.to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=train_cfg.lr, weight_decay=train_cfg.weight_decay)

    # Optional: make torchcfm usage explicit
    ConditionalFlowMatcher = _try_import_torchcfm()
    cfm = None
    if ConditionalFlowMatcher is not None:
        # In torchcfm, the CFM object can sample t and build x_t, but here we keep the rectified target explicit.
        cfm = ConditionalFlowMatcher(sigma=0.0)

    n = x0.shape[0]
    idx = np.arange(n)
    losses: list[float] = []

    for ep in range(int(train_cfg.epochs)):
        np.random.shuffle(idx)
        ep_loss = 0.0
        nb = 0
        for s in range(0, n, int(train_cfg.batch_size)):
            b = idx[s : s + int(train_cfg.batch_size)]
            x0b = torch.from_numpy(x0[b]).to(device)
            x1b = torch.from_numpy(x1[b]).to(device)

            # t ~ U(0,1)
            t = torch.rand(x0b.shape[0], device=device)

            if cfm is not None:
                # use torchcfm to generate x_t with sigma=0 (deterministic interpolation)
                # torchcfm API may require an explicit epsilon argument.
                # With sigma=0.0, epsilon does not affect x_t; pass zeros for compatibility.
                xt = cfm.sample_xt(x0b, x1b, t, torch.zeros_like(x0b))
            else:
                xt = (1.0 - t)[:, None] * x0b + t[:, None] * x1b

            ut = x1b - x0b  # Rectified Flow target
            pred = model(xt, t)

            loss = F.mse_loss(pred, ut)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            ep_loss += float(loss.detach().cpu().item())
            nb += 1
        losses.append(ep_loss / max(1, nb))

    meta = {
        "style": style,
        "num_pairs": int(n),
        "dim": int(D),
        "rf_model": {"hidden": rf_cfg.hidden, "depth": rf_cfg.depth, "dropout": rf_cfg.dropout},
        "train": {"epochs": train_cfg.epochs, "batch_size": train_cfg.batch_size, "lr": train_cfg.lr},
        "used_torchcfm": bool(cfm is not None),
    }
    return RFArtifacts(embedder=embedder, model=model, train_loss=losses, meta=meta)


