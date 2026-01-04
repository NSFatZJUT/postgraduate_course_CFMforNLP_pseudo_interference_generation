from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer

from nlp_pyramidal_flow.utils.lang import detect_lang


def _zh_tokenize(s: str) -> list[str]:
    import jieba

    return list(jieba.cut(s))


@dataclass
class SearchResult:
    doc: str
    score: float
    idx: int


class SimpleRetriever:
    """
    Lightweight retriever:
    - Chinese: BM25 (jieba tokens) as default (robust to mild noise)
    - English: TF-IDF cosine as default
    """

    def __init__(self, docs: list[str]):
        self.docs = docs
        all_text = "\n".join(docs)
        self.lang = detect_lang(all_text, "自动")

        if self.lang == "中文":
            self._tok_docs = [_zh_tokenize(d) for d in docs]
            self._bm25 = BM25Okapi(self._tok_docs)
            self._tfidf = None
        else:
            self._bm25 = None
            self._tfidf_vec = TfidfVectorizer(lowercase=True)
            self._tfidf_mat = self._tfidf_vec.fit_transform(docs)

    def search(self, query: str, top_k: int = 3) -> list[SearchResult]:
        if not self.docs:
            return []
        if self.lang == "中文":
            q = _zh_tokenize(query)
            scores = self._bm25.get_scores(q)
            idxs = np.argsort(scores)[::-1][:top_k]
            return [SearchResult(doc=self.docs[int(i)], score=float(scores[int(i)]), idx=int(i)) for i in idxs]

        qv = self._tfidf_vec.transform([query])
        scores = (self._tfidf_mat @ qv.T).toarray().reshape(-1)
        idxs = np.argsort(scores)[::-1][:top_k]
        return [SearchResult(doc=self.docs[int(i)], score=float(scores[int(i)]), idx=int(i)) for i in idxs]


