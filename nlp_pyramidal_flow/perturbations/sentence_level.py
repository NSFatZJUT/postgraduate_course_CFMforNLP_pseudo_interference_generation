from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from nlp_pyramidal_flow.utils.lang import split_sentences


_ZH_CONNECTIVE_SWAPS = [
    ("因为", "由于"),
    ("所以", "因此"),
    ("但是", "不过"),
    ("同时", "另外"),
]

_ZH_FILLERS = ["其实", "总体来说", "说实话", "简单讲", "某种程度上"]

_EN_CONNECTIVE_SWAPS = [
    ("because", "since"),
    ("therefore", "thus"),
    ("however", "nevertheless"),
    ("also", "additionally"),
]

_EN_FILLERS = ["actually", "overall", "to be honest", "in short", "kind of"]


@dataclass
class SentencePerturb:
    rng: np.random.Generator

    def apply(self, text: str, lang: str, style: str, max_ops: int) -> tuple[str, list[dict]]:
        if max_ops <= 0 or not text.strip():
            return text, []

        ops: list[dict] = []
        out = text

        # op1: connective swaps
        if max_ops > 0:
            if lang == "中文":
                for a, b in self.rng.permutation(_ZH_CONNECTIVE_SWAPS).tolist():
                    if a in out and max_ops > 0:
                        out = out.replace(a, b, 1)
                        ops.append({"name": "connective_swap", "from": a, "to": b})
                        max_ops -= 1
            else:
                lower = out.lower()
                for a, b in self.rng.permutation(_EN_CONNECTIVE_SWAPS).tolist():
                    if a in lower and max_ops > 0:
                        # replace in a case-preserving-ish way: use lower replace, then map back is complex; keep simple
                        idx = lower.find(a)
                        out = out[:idx] + b + out[idx + len(a) :]
                        lower = out.lower()
                        ops.append({"name": "connective_swap", "from": a, "to": b})
                        max_ops -= 1

        # op2: add light filler (style-conditioned)
        if max_ops > 0:
            if style in ("口语化", "社交媒体"):
                filler = self.rng.choice(_ZH_FILLERS if lang == "中文" else _EN_FILLERS)
                out = f"{filler}，{out}" if lang == "中文" else f"{filler}, {out}"
                ops.append({"name": "add_filler", "filler": filler, "pos": "prefix"})
                max_ops -= 1

        # op3: sentence-level light reordering (safe: swap adjacent sentences)
        if max_ops > 0:
            sents = split_sentences(out, lang=lang)
            if len(sents) >= 2:
                i = int(self.rng.integers(0, len(sents) - 1))
                sents[i], sents[i + 1] = sents[i + 1], sents[i]
                joiner = "" if lang == "中文" else " "
                out2 = joiner.join(sents)
                if out2 != out:
                    ops.append({"name": "swap_adjacent_sentences", "i": i, "j": i + 1})
                    out = out2
                    max_ops -= 1

        return out, ops


