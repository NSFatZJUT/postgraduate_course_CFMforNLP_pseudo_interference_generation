from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import jieba
import numpy as np

from nlp_pyramidal_flow.utils.lang import is_protected_index


_ZH_SYNONYM = {
    "取消": ["撤销", "终止", "不再进行"],
    "会议": ["会", "讨论会"],
    "问题": ["麻烦", "困扰", "难题"],
    "鲁棒性": ["稳定性", "抗干扰能力"],
    "增强": ["提升", "加强"],
    "评测": ["测试", "评估"],
}

_EN_SYNONYM = {
    "cancel": ["call off", "abort"],
    "meeting": ["session", "conference"],
    "problem": ["issue", "trouble"],
    "robustness": ["stability", "resilience"],
    "evaluate": ["assess", "test"],
}

_ZH_INSERT = ["真的", "稍微", "大概", "可能", "有点"]
_EN_INSERT = ["really", "slightly", "maybe", "kind of", "probably"]


def _tokenize(text: str, lang: str) -> list[str]:
    if lang == "中文":
        return list(jieba.cut(text))
    # naive English tokenization: keep punctuation as separate tokens
    import re

    return re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE)


def _detokenize(tokens: list[str], lang: str) -> str:
    if lang == "中文":
        return "".join(tokens)
    # insert spaces between words, keep punctuation tight
    out = ""
    for t in tokens:
        if not out:
            out = t
            continue
        if t.isalnum():
            out += " " + t
        else:
            out += t
    return out


@dataclass
class WordPerturb:
    rng: np.random.Generator

    def apply(
        self,
        text: str,
        lang: str,
        style: str,
        max_ops: int,
        protected_spans: Iterable[tuple[int, int, str]] = (),
    ) -> tuple[str, list[dict]]:
        if max_ops <= 0 or not text.strip():
            return text, []

        ops: list[dict] = []
        tokens = _tokenize(text, lang=lang)

        # Build a mapping from token index to (approx) char position start for protection checks
        # This is best-effort; we avoid editing tokens that overlap protected spans.
        pos = 0
        token_starts: list[int] = []
        for t in tokens:
            token_starts.append(pos)
            pos += len(t)

        syn = _ZH_SYNONYM if lang == "中文" else _EN_SYNONYM
        inserts = _ZH_INSERT if lang == "中文" else _EN_INSERT

        # op1: synonym-ish replacement
        idxs = list(range(len(tokens)))
        self.rng.shuffle(idxs)
        for i in idxs:
            if max_ops <= 0:
                break
            t = tokens[i]
            if not t.strip():
                continue
            if is_protected_index(token_starts[i], protected_spans):
                continue
            key = t if lang == "中文" else t.lower()
            if key in syn:
                rep = str(self.rng.choice(syn[key]))
                tokens[i] = rep
                ops.append({"name": "synonym_replace", "i": i, "from": t, "to": rep})
                max_ops -= 1

        # op2: light insertion (style conditioned)
        if max_ops > 0 and style in ("口语化", "社交媒体"):
            # insert near the beginning but not inside protected spans
            i = int(self.rng.integers(0, max(1, min(5, len(tokens)))))
            if len(tokens) > 0 and not is_protected_index(token_starts[min(i, len(token_starts) - 1)], protected_spans):
                w = str(self.rng.choice(inserts))
                tokens.insert(i, w)
                ops.append({"name": "insert_filler_word", "i": i, "word": w})
                max_ops -= 1

        return _detokenize(tokens, lang=lang), ops


