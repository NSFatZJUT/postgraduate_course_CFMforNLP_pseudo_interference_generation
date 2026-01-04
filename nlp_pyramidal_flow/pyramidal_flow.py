from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from nlp_pyramidal_flow.config import PIGConfig
from nlp_pyramidal_flow.perturbations.char_level import CharNoise
from nlp_pyramidal_flow.perturbations.sentence_level import SentencePerturb
from nlp_pyramidal_flow.perturbations.subword_level import SubwordPerturb
from nlp_pyramidal_flow.perturbations.word_level import WordPerturb
from nlp_pyramidal_flow.utils.lang import detect_lang, protect_spans


@dataclass
class FlowOp:
    level: str
    name: str
    detail: dict[str, Any]


@dataclass
class FlowTrace:
    lang: str
    style: str
    ops: list[FlowOp]

    def to_dict(self) -> dict[str, Any]:
        return {
            "lang": self.lang,
            "style": self.style,
            "num_ops": len(self.ops),
            "ops": [
                {"level": o.level, "name": o.name, "detail": o.detail}
                for o in self.ops
            ],
        }


@dataclass
class PerturbResult:
    text: str
    trace: FlowTrace


class PyramidalFlowPerturber:
    """
    NLP version of "pyramidal flow" pseudo-interference generator:
    Apply perturbations in a coarse-to-fine pyramid, recording a trace.
    """

    def __init__(self, cfg: PIGConfig):
        self.cfg = cfg
        self._rng = np.random.default_rng(cfg.seed)

        self._sentence = SentencePerturb(self._rng)
        self._word = WordPerturb(self._rng)
        self._subword = SubwordPerturb(self._rng)
        self._char = CharNoise(self._rng)

    def perturb(
        self,
        text: str,
        style: str = "通用",
        extra_protected_spans: Optional[list[tuple[int, int, str]]] = None,
    ) -> PerturbResult:
        lang = detect_lang(text, self.cfg.lang_hint)
        spans = protect_spans(text) if self.cfg.protect_numbers else []
        if extra_protected_spans:
            spans = sorted(list(spans) + list(extra_protected_spans))
        trace = FlowTrace(lang=lang, style=style, ops=[])

        remaining = int(self.cfg.max_ops)
        out = text

        def _budget_take(k: int) -> int:
            nonlocal remaining
            k = max(0, min(int(k), remaining))
            remaining -= k
            return k

        # Coarse-to-fine allocation. Even if a layer gets 0, we still allow next layers.
        k_sent = _budget_take(int(round(self.cfg.max_ops * self.cfg.intensity_sentence)))
        k_word = _budget_take(int(round(self.cfg.max_ops * self.cfg.intensity_word)))
        k_subw = _budget_take(int(round(self.cfg.max_ops * self.cfg.intensity_subword)))
        k_char = _budget_take(remaining)

        out, ops = self._sentence.apply(out, lang=lang, style=style, max_ops=k_sent)
        trace.ops.extend([FlowOp(level="sentence", name=o["name"], detail=o) for o in ops])

        out, ops = self._word.apply(out, lang=lang, style=style, max_ops=k_word, protected_spans=spans)
        trace.ops.extend([FlowOp(level="word", name=o["name"], detail=o) for o in ops])

        out, ops = self._subword.apply(out, lang=lang, style=style, max_ops=k_subw, protected_spans=spans)
        trace.ops.extend([FlowOp(level="subword", name=o["name"], detail=o) for o in ops])

        out, ops = self._char.apply(out, lang=lang, style=style, max_ops=k_char, protected_spans=spans)
        trace.ops.extend([FlowOp(level="char", name=o["name"], detail=o) for o in ops])

        return PerturbResult(text=out, trace=trace)


