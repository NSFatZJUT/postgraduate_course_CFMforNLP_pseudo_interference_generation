from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np

from nlp_pyramidal_flow.tasks.matching import cosine_match


@dataclass
class QualityScores:
    semantic_retention: float  # 0~1, higher better
    fluency: float  # 0~1, higher better
    effectiveness: float  # 0~1, higher means larger degradation (for single-sample proxy)
    passed_semantic: bool
    passed_fluency: bool
    effectiveness_band: str  # "too_weak"/"optimal"/"too_strong"

    def to_dict(self) -> dict:
        return {
            "semantic_retention": float(self.semantic_retention),
            "fluency": float(self.fluency),
            "effectiveness": float(self.effectiveness),
            "passed_semantic": bool(self.passed_semantic),
            "passed_fluency": bool(self.passed_fluency),
            "effectiveness_band": self.effectiveness_band,
        }


def _heuristic_fluency(text: str) -> float:
    """
    Lightweight fluency proxy (no model download):
    - penalize extreme repetition / weird whitespace
    - penalize too many non-word symbols
    Output 0~1.
    """
    s = text.strip()
    if not s:
        return 0.0

    n = len(s)
    rep = 0
    cur = 1
    for i in range(1, n):
        if s[i] == s[i - 1]:
            cur += 1
            rep = max(rep, cur)
        else:
            cur = 1
    rep_pen = min(0.6, max(0.0, (rep - 3) * 0.1))

    ws_pen = 0.0
    if "  " in s or "\t" in s or "\n" in s:
        ws_pen = 0.15

    sym = sum(1 for ch in s if not ch.isalnum() and ch not in "，。！？!?.,;:（）()【】[]-_@/ ")
    sym_ratio = sym / max(1, n)
    sym_pen = min(0.3, sym_ratio * 1.5)

    base = 1.0 - (rep_pen + ws_pen + sym_pen)
    return float(max(0.0, min(1.0, base)))


class GPT2FluencyScorer:
    """
    Optional: use (distil)GPT2 perplexity to score fluency.
    Returns 0~1 where higher is better.
    """

    def __init__(self, model_name: str = "distilgpt2", device: str = "cpu"):
        self.model_name = model_name
        self.device = device
        self._tok = None
        self._model = None

    def _lazy_load(self):
        if self._tok is not None:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._tok = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModelForCausalLM.from_pretrained(self.model_name)
        self._model.to(self.device)
        self._model.eval()

    def score(self, text: str) -> float:
        s = text.strip()
        if not s:
            return 0.0
        self._lazy_load()
        import torch

        enc = self._tok(s, return_tensors="pt", truncation=True, max_length=256)
        enc = {k: v.to(self.device) for k, v in enc.items()}
        with torch.no_grad():
            out = self._model(**enc, labels=enc["input_ids"])
            loss = float(out.loss.detach().cpu().item())
        ppl = float(np.exp(min(20.0, loss)))  # cap to avoid overflow
        # map perplexity roughly into 0~1; ppl 10 -> ~0.8, ppl 50 -> ~0.4
        score = 1.0 / (1.0 + np.log1p(ppl))
        return float(max(0.0, min(1.0, score)))


def score_quality(
    original: str,
    perturbed: str,
    classifier: Optional[Any] = None,
    fluency_scorer: Optional[Any] = None,
    semantic_embedder: Optional[Any] = None,
    semantic_threshold: float = 0.7,
    fluency_threshold: float = 0.8,
) -> QualityScores:
    """
    Quality scoring in 3 dimensions:
    - semantic retention: cosine_match(original, perturbed)
    - fluency: GPT2-based if provided, else heuristic
    - effectiveness: proxy by classifier confidence drop if classifier provided
    """
    if semantic_embedder is not None:
        try:
            v = semantic_embedder.encode([original, perturbed])
            sem = float(np.dot(v[0], v[1]) / (np.linalg.norm(v[0]) * np.linalg.norm(v[1]) + 1e-12))
            sem = float(max(0.0, min(1.0, sem)))
        except Exception:
            sem = float(cosine_match(original, perturbed))
    else:
        sem = float(cosine_match(original, perturbed))
    if fluency_scorer is not None:
        try:
            flu = float(fluency_scorer.score(perturbed))
        except Exception:
            flu = float(_heuristic_fluency(perturbed))
    else:
        flu = float(_heuristic_fluency(perturbed))

    eff = 0.0
    if classifier is not None:
        try:
            p0 = float(np.max(classifier.predict_proba([original])[0]))
            p1 = float(np.max(classifier.predict_proba([perturbed])[0]))
            eff = float(max(0.0, min(1.0, p0 - p1)))
        except Exception:
            eff = 0.0

    band = "too_weak"
    if 0.1 <= eff <= 0.3:
        band = "optimal"
    elif eff > 0.3:
        band = "too_strong"

    return QualityScores(
        semantic_retention=sem,
        fluency=flu,
        effectiveness=eff,
        passed_semantic=sem >= semantic_threshold,
        passed_fluency=flu >= fluency_threshold,
        effectiveness_band=band,
    )


