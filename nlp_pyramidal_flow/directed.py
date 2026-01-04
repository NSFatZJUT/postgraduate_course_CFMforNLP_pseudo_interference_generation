from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import re

from nlp_pyramidal_flow.tasks.ner import simple_ner
from nlp_pyramidal_flow.utils.lang import detect_lang


Span = Tuple[int, int, str]


@dataclass
class DirectedConfig:
    mode: str = "通用"  # 通用 / NER-保护实体 / 匹配-保留关键词 / QA-改写问题
    keyword_topk: int = 6


def spans_for_ner_protect_entities(text: str) -> List[Span]:
    ents = simple_ner(text)
    spans: List[Span] = []
    for e in ents:
        if e["label"] in ("PER", "ORG", "LOC", "EMAIL", "URL", "DATE", "NUMBER"):
            spans.append((int(e["start"]), int(e["end"]), f"entity:{e['label']}"))
    return sorted(spans)


def spans_for_match_protect_keywords(text: str, topk: int = 6) -> List[Span]:
    """
    Protect keyword-like spans to keep core semantics stable.
    Heuristic:
    - Chinese: protect 2+ length CJK runs and numbers/emails/urls
    - English: protect 4+ length words
    """
    lang = detect_lang(text, "自动")
    spans: List[Span] = []

    if lang == "中文":
        # pick longer CJK runs
        for m in re.finditer(r"[\u4e00-\u9fff]{2,6}", text):
            spans.append((m.start(), m.end(), "kw"))
    else:
        for m in re.finditer(r"\b[a-zA-Z]{4,}\b", text):
            spans.append((m.start(), m.end(), "kw"))

    # keep only topk by length (and unique by start/end)
    spans = sorted(spans, key=lambda s: (-(s[1] - s[0]), s[0]))
    out: List[Span] = []
    seen = set()
    for a, b, k in spans:
        if (a, b) in seen:
            continue
        out.append((a, b, k))
        seen.add((a, b))
        if len(out) >= int(topk):
            break
    return sorted(out)


def spans_for_qa_question(text: str) -> List[Span]:
    """
    For QA question rewriting, preserve numbers/entities to keep answer core stable.
    """
    spans = spans_for_ner_protect_entities(text)
    # also preserve quoted strings
    for m in re.finditer(r"“[^”]{1,30}”|\"[^\"]{1,30}\"", text):
        spans.append((m.start(), m.end(), "quote"))
    return sorted(spans)


def compute_directed_spans(text: str, cfg: DirectedConfig) -> List[Span]:
    if cfg.mode == "NER-保护实体":
        return spans_for_ner_protect_entities(text)
    if cfg.mode == "匹配-保留关键词":
        return spans_for_match_protect_keywords(text, topk=cfg.keyword_topk)
    if cfg.mode == "QA-改写问题":
        return spans_for_qa_question(text)
    return []


