from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from pypinyin import lazy_pinyin

from nlp_pyramidal_flow.utils.lang import is_protected_index


_ZH_OCR_CONFUSION = {
    "日": ["目"],
    "目": ["日"],
    "未": ["末"],
    "末": ["未"],
    "士": ["土"],
    "土": ["士"],
    "口": ["囗"],
    "囗": ["口"],
    "己": ["已", "巳"],
    "已": ["己"],
    "人": ["入"],
    "入": ["人"],
}

# Tiny homophone candidate set (expandable for coursework)
_ZH_PINYIN_CAND = {
    "shi": list("是事市式试识时十使"),
    "de": list("的得德"),
    "li": list("里理力利立例礼离"),
    "zhang": list("张章长涨掌"),
    "wang": list("王往网望忘"),
}

_EN_KEYBOARD_NEIGHBORS = {
    "a": "qwsz",
    "s": "qwedxza",
    "d": "wersfxc",
    "f": "ertdgcv",
    "g": "rtyfhvb",
    "h": "tyugjbn",
    "j": "yuihknm",
    "k": "uiojlm",
    "l": "opk",
    "e": "wsdr",
    "r": "edft",
    "t": "rfgy",
    "y": "tghu",
    "u": "yhj",
    "i": "ujk",
    "o": "ikl",
    "n": "bhjm",
    "m": "njk",
}


@dataclass
class CharNoise:
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
        out = list(text)

        # pick editable positions (skip whitespace & protected spans)
        idxs = [
            i
            for i, ch in enumerate(out)
            if ch.strip() and not is_protected_index(i, protected_spans)
        ]
        if not idxs:
            return text, []

        # NOTE:
        # We may insert/pop characters below, which changes list length and shifts indices.
        # If we iterate indices in arbitrary order, we can hit IndexError (stale indices).
        # Processing from high -> low ensures edits at higher positions don't invalidate
        # the remaining lower indices (and also keeps protected spans safer).
        self.rng.shuffle(idxs)
        idxs = sorted(idxs, reverse=True)
        for i in idxs:
            if max_ops <= 0:
                break
            if i < 0 or i >= len(out):
                # out length may have changed due to earlier edits; skip stale index safely
                continue
            ch = out[i]

            # choose an operator depending on style
            if lang == "中文":
                if style == "OCR风格" and ch in _ZH_OCR_CONFUSION:
                    rep = str(self.rng.choice(_ZH_OCR_CONFUSION[ch]))
                    out[i] = rep
                    ops.append({"name": "ocr_confusion", "pos": i, "from": ch, "to": rep})
                    max_ops -= 1
                    continue

                # homophone-ish: use pinyin to pick a candidate char
                py = lazy_pinyin(ch)[0] if ch.strip() else ""
                cand = _ZH_PINYIN_CAND.get(py)
                if cand and ch in cand and len(cand) > 1:
                    rep = str(self.rng.choice([c for c in cand if c != ch]))
                    out[i] = rep
                    ops.append({"name": "pinyin_homophone", "pos": i, "from": ch, "to": rep, "pinyin": py})
                    max_ops -= 1
                    continue

                # generic: duplicate or delete
                if self.rng.random() < 0.5:
                    out.insert(i, ch)
                    ops.append({"name": "char_duplicate", "pos": i, "char": ch})
                else:
                    out.pop(i)
                    ops.append({"name": "char_delete", "pos": i, "char": ch})
                max_ops -= 1
            else:
                low = ch.lower()
                if style in ("ASR风格", "社交媒体") and low in _EN_KEYBOARD_NEIGHBORS:
                    rep = str(self.rng.choice(list(_EN_KEYBOARD_NEIGHBORS[low])))
                    out[i] = rep if ch.islower() else rep.upper()
                    ops.append({"name": "keyboard_neighbor", "pos": i, "from": ch, "to": out[i]})
                    max_ops -= 1
                    continue
                # generic typo: swap case or repeat
                if ch.isalpha() and self.rng.random() < 0.5:
                    out[i] = ch.swapcase()
                    ops.append({"name": "swap_case", "pos": i, "from": ch, "to": out[i]})
                else:
                    out.insert(i, ch)
                    ops.append({"name": "char_repeat", "pos": i, "char": ch})
                max_ops -= 1

        return "".join(out), ops


