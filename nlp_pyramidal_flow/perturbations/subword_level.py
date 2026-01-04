from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from nlp_pyramidal_flow.utils.lang import is_protected_index


_ZH_FULLWIDTH_MAP = {
    "/": "／",
    ":": "：",
    ",": "，",
    ".": "。",
    "!": "！",
    "?": "？",
    "(": "（",
    ")": "）",
}

_ZH_HALFWIDTH_MAP = {v: k for k, v in _ZH_FULLWIDTH_MAP.items()}

_SOCIAL_DUP_PUNCT = ["!!", "!!!", "？？", "!!??", "……", "。。。"]


@dataclass
class SubwordPerturb:
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
        out = text

        # op1: fullwidth/halfwidth punctuation swap (tokenization sensitivity)
        if max_ops > 0 and lang == "中文":
            candidates = []
            for i, ch in enumerate(out):
                if is_protected_index(i, protected_spans):
                    continue
                if ch in _ZH_FULLWIDTH_MAP or ch in _ZH_HALFWIDTH_MAP:
                    candidates.append(i)
            if candidates:
                i = int(self.rng.choice(candidates))
                ch = out[i]
                if ch in _ZH_FULLWIDTH_MAP:
                    rep = _ZH_FULLWIDTH_MAP[ch]
                else:
                    rep = _ZH_HALFWIDTH_MAP[ch]
                out = out[:i] + rep + out[i + 1 :]
                ops.append({"name": "punct_fullwidth_swap", "pos": i, "from": ch, "to": rep})
                max_ops -= 1

        # op2: whitespace jitter (subword boundary changes)
        if max_ops > 0:
            # insert or remove a single space around a random position (avoid protected)
            idxs = [i for i in range(len(out)) if not is_protected_index(i, protected_spans)]
            if idxs:
                i = int(self.rng.choice(idxs))
                if out[i] == " ":
                    out = out[:i] + out[i + 1 :]
                    ops.append({"name": "whitespace_remove", "pos": i})
                else:
                    out = out[:i] + " " + out[i:]
                    ops.append({"name": "whitespace_insert", "pos": i})
                max_ops -= 1

        # op3: social media punctuation duplication
        if max_ops > 0 and style in ("社交媒体", "口语化"):
            punct = str(self.rng.choice(_SOCIAL_DUP_PUNCT))
            out = out + punct
            ops.append({"name": "append_social_punct", "punct": punct})
            max_ops -= 1

        return out, ops


