from __future__ import annotations

import re

from nlp_pyramidal_flow.tasks.ner import simple_ner
from nlp_pyramidal_flow.utils.lang import detect_lang


_ZH_POS_MAP = {
    "不错": "挺不错",
    "很好": "非常好",
    "满意": "很满意",
    "喜欢": "挺喜欢",
}

_ZH_NEG_MAP = {
    "一般": "有点一般",
    "不好": "不太好",
    "差": "有点差",
    "失望": "挺失望",
    "糟糕": "挺糟糕",
}


def apply_style(text: str, style: str) -> str:
    """
    Post-edit a generated text to match a user-selected "fun" style.
    This keeps the system lightweight and makes the effect observable.
    """
    if not text.strip():
        return text

    lang = detect_lang(text, "自动")
    out = text

    if style == "默认":
        return out

    if style == "口语化":
        if lang == "中文":
            # add mild colloquial markers without destroying core info
            if not out.startswith(("说实话", "其实", "老实说")):
                out = "说实话，" + out
            out = out.replace("。", "！").replace("，", "，就")
            if not out.endswith(("～", "!", "！")):
                out += "～"
        else:
            out = "Honestly, " + out
        return out

    if style == "轻微消极":
        if lang == "中文":
            for a, b in _ZH_NEG_MAP.items():
                out = out.replace(a, b)
            if "有点" not in out and "不太" not in out:
                out = "有点… " + out
        return out

    if style == "轻微积极":
        if lang == "中文":
            for a, b in _ZH_POS_MAP.items():
                out = out.replace(a, b)
            if "挺" not in out and "很" not in out:
                out = "挺好： " + out
        return out

    if style == "实体模糊":
        ents = simple_ner(out)
        # replace from back to front to keep indices valid
        for e in sorted(ents, key=lambda x: x["start"], reverse=True):
            lab = e["label"]
            repl = None
            if lab == "PER":
                repl = "某用户"
            elif lab == "ORG":
                repl = "某机构"
            elif lab == "LOC":
                repl = "某地"
            elif lab in ("EMAIL", "URL"):
                repl = "（已隐藏）"
            if repl is None:
                continue
            out = out[: e["start"]] + repl + out[e["end"] :]
        # also soften explicit names like 张三/李四
        out = re.sub(r"(张三|李四|王五)", "某用户", out)
        return out

    return out


