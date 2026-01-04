from __future__ import annotations

import re

from nlp_pyramidal_flow.utils.lang import detect_lang


def simple_ner(text: str) -> list[dict]:
    """
    A heuristic NER module for demo/report purposes (no heavy model dependency).
    Returns a list of {text,label,start,end}.

    Notes:
    - English: capitalized spans as PERSON/ORG-ish
    - Chinese: suffix-based ORG/LOC; emails/urls/dates/numbers by regex
    """
    lang = detect_lang(text, "自动")
    ents: list[dict] = []

    def add(m: re.Match, label: str) -> None:
        ents.append({"text": m.group(0), "label": label, "start": m.start(), "end": m.end()})

    for m in re.finditer(r"https?://\S+", text):
        add(m, "URL")
    for m in re.finditer(r"[\w.+-]+@[\w-]+\.[\w.-]+", text):
        add(m, "EMAIL")
    for m in re.finditer(r"\b\d{4}[/-]\d{1,2}[/-]\d{1,2}\b", text):
        add(m, "DATE")
    for m in re.finditer(r"\b\d+(?:\.\d+)?\b", text):
        add(m, "NUMBER")

    if lang == "中文":
        org_suffix = r"(大学|公司|集团|银行|研究院|学院|医院|委员会)"
        loc_suffix = r"(省|市|县|区|镇|乡|街道|路)"

        for m in re.finditer(rf"[\u4e00-\u9fff]{{2,10}}{org_suffix}", text):
            add(m, "ORG")
        for m in re.finditer(rf"[\u4e00-\u9fff]{{2,10}}{loc_suffix}", text):
            add(m, "LOC")

        # very light PERSON heuristic: 2-3 Chinese chars around "的/在/对/给/与"
        for m in re.finditer(r"(?:(?<=的)|(?<=在)|(?<=对)|(?<=给)|(?<=与))([\u4e00-\u9fff]{2,3})", text):
            ents.append({"text": m.group(1), "label": "PER", "start": m.start(1), "end": m.end(1)})
    else:
        for m in re.finditer(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b", text):
            span = m.group(1)
            label = "PER" if len(span.split()) <= 2 else "ORG"
            ents.append({"text": span, "label": label, "start": m.start(1), "end": m.end(1)})

    # de-dup by (start,end,label)
    seen = set()
    out = []
    for e in sorted(ents, key=lambda x: (x["start"], x["end"], x["label"])):
        k = (e["start"], e["end"], e["label"])
        if k not in seen:
            out.append(e)
            seen.add(k)
    return out


