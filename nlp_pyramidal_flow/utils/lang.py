import re
from typing import Iterable


_CJK_RE = re.compile(r"[\u4e00-\u9fff]")
_WS_RE = re.compile(r"\s+")


def detect_lang(text: str, lang_hint: str = "自动") -> str:
    if lang_hint in ("中文", "英文"):
        return lang_hint
    if _CJK_RE.search(text):
        return "中文"
    return "英文"


def normalize_ws(text: str) -> str:
    return _WS_RE.sub(" ", text).strip()


def split_sentences(text: str, lang: str) -> list[str]:
    text = text.strip()
    if not text:
        return []
    if lang == "中文":
        parts = re.split(r"([。！？!?])", text)
        out: list[str] = []
        buf = ""
        for p in parts:
            if p in ("。", "！", "？", "!", "?"):
                buf += p
                out.append(buf)
                buf = ""
            else:
                buf += p
        if buf.strip():
            out.append(buf.strip())
        return [s.strip() for s in out if s.strip()]
    # English-ish
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [p.strip() for p in parts if p.strip()]


def protect_spans(text: str) -> list[tuple[int, int, str]]:
    """
    Return protected spans (start, end, kind) that should be avoided during perturbation.
    """
    spans: list[tuple[int, int, str]] = []
    for m in re.finditer(r"https?://\S+", text):
        spans.append((m.start(), m.end(), "url"))
    for m in re.finditer(r"[\w.+-]+@[\w-]+\.[\w.-]+", text):
        spans.append((m.start(), m.end(), "email"))
    for m in re.finditer(r"\b\d+(?:[.,:/-]\d+)*\b", text):
        spans.append((m.start(), m.end(), "number"))
    spans.sort()
    return spans


def is_protected_index(i: int, spans: Iterable[tuple[int, int, str]]) -> bool:
    for a, b, _ in spans:
        if a <= i < b:
            return True
    return False


