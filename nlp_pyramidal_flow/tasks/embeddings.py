from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from nlp_pyramidal_flow.utils.lang import detect_lang


def _zh_tokenizer(s: str) -> list[str]:
    import jieba

    return list(jieba.cut(s))


@dataclass
class TfidfEmbedder:
    vectorizer: TfidfVectorizer
    lang: str

    @classmethod
    def fit(cls, texts: list[str], lang_hint: str = "自动") -> "TfidfEmbedder":
        lang = detect_lang("\n".join(texts[:50]), lang_hint)
        if lang == "中文":
            vec = TfidfVectorizer(tokenizer=_zh_tokenizer, lowercase=False, min_df=1)
        else:
            vec = TfidfVectorizer(lowercase=True, min_df=1)
        vec.fit(texts)
        return cls(vectorizer=vec, lang=lang)

    def transform(self, texts: list[str]) -> np.ndarray:
        X = self.vectorizer.transform(texts)
        return X.toarray().astype(np.float32)


