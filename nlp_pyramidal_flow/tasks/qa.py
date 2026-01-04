from __future__ import annotations

from dataclasses import dataclass

from rapidfuzz import fuzz

from nlp_pyramidal_flow.tasks.retrieval import SimpleRetriever
from nlp_pyramidal_flow.utils.lang import detect_lang, split_sentences


@dataclass
class QAResult:
    answer: str
    top_doc: str
    score: float


def retrieval_qa(question: str, retriever: SimpleRetriever) -> QAResult:
    """
    Retrieval-based QA:
    1) retrieve top doc
    2) pick the best-matching sentence as answer
    """
    hits = retriever.search(question, top_k=1)
    if not hits:
        return QAResult(answer="（知识库为空或未命中）", top_doc="", score=0.0)

    top = hits[0].doc
    lang = detect_lang(question + top, "自动")
    sents = split_sentences(top, lang=lang)
    if not sents:
        return QAResult(answer=top, top_doc=top, score=float(hits[0].score))

    best = max(sents, key=lambda s: fuzz.partial_ratio(question, s))
    sc = fuzz.partial_ratio(question, best) / 100.0
    return QAResult(answer=best, top_doc=top, score=sc)


