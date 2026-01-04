from __future__ import annotations

import numpy as np
from rapidfuzz import fuzz
from sklearn.feature_extraction.text import TfidfVectorizer

from nlp_pyramidal_flow.utils.lang import detect_lang


def _zh_tokenizer(s: str) -> list[str]:
    import jieba

    return list(jieba.cut(s))


def cosine_match(a: str, b: str) -> float:
    """
    A lightweight matching score:
    - Use TF-IDF cosine in a shared small corpus
    - Fallback mix-in: partial ratio (robust to small noise)
    """
    lang = detect_lang(a + b, "自动")
    if lang == "中文":
        vec = TfidfVectorizer(tokenizer=_zh_tokenizer, lowercase=False)
    else:
        vec = TfidfVectorizer(lowercase=True)
    X = vec.fit_transform([a, b])
    v0 = X[0].toarray()[0]
    v1 = X[1].toarray()[0]
    denom = float(np.linalg.norm(v0) * np.linalg.norm(v1) + 1e-12)
    cos = float(np.dot(v0, v1) / denom)
    pr = fuzz.partial_ratio(a, b) / 100.0
    return float(0.7 * cos + 0.3 * pr)


