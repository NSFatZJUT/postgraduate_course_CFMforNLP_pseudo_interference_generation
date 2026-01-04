from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


def _require_transformers():
    try:
        import transformers  # noqa: F401
    except Exception as e:
        raise RuntimeError("缺少 transformers 依赖：请先安装 requirements.txt。") from e


@dataclass
class TransformerEmbedder:
    model_name: str
    device: str = "cpu"
    max_length: int = 256

    _tok: Optional[object] = None
    _model: Optional[object] = None

    def _lazy_load(self):
        if self._tok is not None:
            return
        _require_transformers()
        import torch
        from transformers import AutoModel, AutoTokenizer

        self._tok = AutoTokenizer.from_pretrained(self.model_name, use_fast=True)
        self._model = AutoModel.from_pretrained(self.model_name)
        self._model.to(self.device)
        self._model.eval()

    def encode(self, texts: list[str]) -> np.ndarray:
        self._lazy_load()
        import torch

        tok = self._tok
        model = self._model

        enc = tok(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=int(self.max_length),
        )
        enc = {k: v.to(self.device) for k, v in enc.items()}
        with torch.no_grad():
            out = model(**enc)
            last = out.last_hidden_state  # [B, T, H]
            mask = enc.get("attention_mask", None)
            if mask is None:
                pooled = last.mean(dim=1)
            else:
                m = mask.unsqueeze(-1).type_as(last)
                pooled = (last * m).sum(dim=1) / (m.sum(dim=1) + 1e-12)
        vec = pooled.detach().cpu().numpy().astype(np.float32)
        # L2 normalize
        denom = np.linalg.norm(vec, axis=1, keepdims=True) + 1e-12
        return vec / denom


