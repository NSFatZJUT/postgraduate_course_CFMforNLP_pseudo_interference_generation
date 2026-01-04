from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline

from nlp_pyramidal_flow.pyramidal_flow import PyramidalFlowPerturber
from nlp_pyramidal_flow.utils.lang import detect_lang


def _zh_tokenizer(s: str) -> list[str]:
    import jieba

    return list(jieba.cut(s))


def _build_vectorizer(lang: str) -> TfidfVectorizer:
    if lang == "中文":
        return TfidfVectorizer(tokenizer=_zh_tokenizer, lowercase=False, min_df=1)
    return TfidfVectorizer(lowercase=True, min_df=1)


@dataclass
class ClassificationTrainer:
    pipe: Any = None  # Pipeline, but keep 3.8+ friendly typing
    labels_: Any = None  # list[str] when available
    lang_: Any = None  # str when available

    def fit(self, df: pd.DataFrame) -> None:
        df = df.dropna(subset=["text", "label"])
        x = df["text"].astype(str).tolist()
        y = df["label"].astype(str).tolist()
        self.labels_ = sorted(list(set(y)))
        self.lang_ = detect_lang("\n".join(x[:50]), "自动")
        vec = _build_vectorizer(self.lang_)
        clf = LogisticRegression(max_iter=200, n_jobs=1)
        self.pipe = Pipeline([("tfidf", vec), ("clf", clf)])
        self.pipe.fit(x, y)

    def fit_with_augmentation(
        self,
        df: pd.DataFrame,
        perturber: PyramidalFlowPerturber,
        style: str,
        aug_ratio: float = 0.5,
    ) -> None:
        df = df.dropna(subset=["text", "label"]).copy()
        df["text"] = df["text"].astype(str)
        n = len(df)
        if n == 0:
            raise ValueError("数据为空：请提供 text,label 列。")

        rng = np.random.default_rng(perturber.cfg.seed + 7)
        idx = np.arange(n)
        rng.shuffle(idx)
        k = int(round(n * float(aug_ratio)))
        aug_idx = set(idx[:k].tolist())

        texts = []
        labels = []
        for i, row in df.reset_index(drop=True).iterrows():
            x = row["text"]
            if i in aug_idx:
                x = perturber.perturb(x, style=style).text
            texts.append(x)
            labels.append(str(row["label"]))

        self.labels_ = sorted(list(set(labels)))
        self.lang_ = detect_lang("\n".join(texts[:50]), "自动")
        vec = _build_vectorizer(self.lang_)
        clf = LogisticRegression(max_iter=200, n_jobs=1)
        self.pipe = Pipeline([("tfidf", vec), ("clf", clf)])
        self.pipe.fit(texts, labels)

    def predict(self, text_list: list[str]) -> list[str]:
        if self.pipe is None:
            raise RuntimeError("模型未训练。")
        return self.pipe.predict(text_list).tolist()

    def predict_proba(self, text_list: list[str]) -> np.ndarray:
        if self.pipe is None:
            raise RuntimeError("模型未训练。")
        clf: LogisticRegression = self.pipe.named_steps["clf"]
        return clf.predict_proba(self.pipe.named_steps["tfidf"].transform(text_list))

    def greedy_attack(self, text: str, perturber: PyramidalFlowPerturber, style: str = "通用") -> dict[str, Any]:
        """
        A lightweight adversarial attack for the TF-IDF+LR classifier:
        - find important tokens for current prediction
        - try to corrupt them (space injection / char corruption) to flip prediction
        """
        if self.pipe is None:
            raise RuntimeError("模型未训练。")

        vec: TfidfVectorizer = self.pipe.named_steps["tfidf"]
        clf: LogisticRegression = self.pipe.named_steps["clf"]

        x = text
        trace: list[dict] = []
        pred0 = self.pipe.predict([x])[0]
        proba0 = float(np.max(self.predict_proba([x])[0]))

        budget = max(1, int(perturber.cfg.max_ops // 2))

        for step in range(budget):
            X = vec.transform([x])
            feat_names = np.array(vec.get_feature_names_out())
            # contribution for predicted class
            class_idx = int(np.where(clf.classes_ == pred0)[0][0])
            # sklearn LogisticRegression:
            # - multiclass: coef_.shape == [C, D]
            # - binary:     coef_.shape == [1, D] and corresponds to classes_[1]
            if getattr(clf, "coef_", None) is None:
                break
            if clf.coef_.shape[0] == 1:
                coef = clf.coef_[0] if class_idx == 1 else -clf.coef_[0]
            else:
                coef = clf.coef_[class_idx]
            contrib = X.toarray().reshape(-1) * coef
            # pick top contributing feature present in x
            if contrib.size == 0 or float(np.max(np.abs(contrib))) <= 0:
                break
            top_j = int(np.argmax(contrib))
            tok = str(feat_names[top_j])
            if not tok.strip():
                break

            x2, op = _corrupt_token(x, tok, lang=self.lang_ or "中文")
            if x2 == x:
                # fallback: random pyramidal perturb
                pr = perturber.perturb(x, style=style)
                x2 = pr.text
                op = {"name": "fallback_pyramidal", "token": tok, "inner_ops": pr.trace.to_dict()}

            trace.append({"step": step, "target_token": tok, "op": op})
            x = x2

            pred = self.pipe.predict([x])[0]
            proba = float(np.max(self.predict_proba([x])[0]))
            if pred != pred0:
                return {
                    "adv_text": x,
                    "trace": trace,
                    "summary": {
                        "orig_pred": pred0,
                        "orig_conf": proba0,
                        "adv_pred": pred,
                        "adv_conf": proba,
                        "steps": step + 1,
                        "success": True,
                    },
                }

        pred = self.pipe.predict([x])[0]
        proba = float(np.max(self.predict_proba([x])[0]))
        return {
            "adv_text": x,
            "trace": trace,
            "summary": {
                "orig_pred": pred0,
                "orig_conf": proba0,
                "adv_pred": pred,
                "adv_conf": proba,
                "steps": len(trace),
                "success": pred != pred0,
            },
        }


def _corrupt_token(text: str, token: str, lang: str) -> tuple[str, dict]:
    """
    Targeted corruption for a specific token occurrence.
    Keeps it simple and deterministic-ish for demo.
    """
    i = text.find(token)
    if i < 0:
        # try case-insensitive for English
        if lang != "中文":
            low = text.lower()
            j = low.find(token.lower())
            if j >= 0:
                i = j
                token = text[j : j + len(token)]
    if i < 0:
        return text, {"name": "no_op", "reason": "token_not_found"}

    span = text[i : i + len(token)]

    # subword-like: insert a space inside the token to break tokenization
    if len(span) >= 3:
        k = i + max(1, len(span) // 2)
        out = text[:k] + " " + text[k:]
        return out, {"name": "space_inject", "pos": k, "token": span}

    # char-like: duplicate or swapcase
    if span and span[0].isalpha():
        rep = span[0].swapcase() + span[1:]
        out = text[:i] + rep + text[i + len(span) :]
        return out, {"name": "swapcase_char", "pos": i, "from": span, "to": rep}

    if span:
        out = text[:i] + span + span + text[i + len(span) :]
        return out, {"name": "duplicate_token", "pos": i, "token": span}

    return text, {"name": "no_op", "reason": "empty_span"}


def evaluate_classifier(
    df: pd.DataFrame,
    trainer: ClassificationTrainer,
    perturber: PyramidalFlowPerturber,
    style: str,
) -> dict[str, Any]:
    df = df.dropna(subset=["text", "label"]).copy()
    df["text"] = df["text"].astype(str)
    df["label"] = df["label"].astype(str)
    x = df["text"].tolist()
    y = df["label"].tolist()

    y_pred = trainer.predict(x)
    acc = float(accuracy_score(y, y_pred))
    f1 = float(f1_score(y, y_pred, average="macro"))

    x_noisy = [perturber.perturb(t, style=style).text for t in x]
    y_pred_n = trainer.predict(x_noisy)
    acc_n = float(accuracy_score(y, y_pred_n))
    f1_n = float(f1_score(y, y_pred_n, average="macro"))

    # worst examples: correct->wrong or confidence drop
    proba_c = np.max(trainer.predict_proba(x), axis=1)
    proba_n = np.max(trainer.predict_proba(x_noisy), axis=1)
    drop = proba_c - proba_n
    wrong_flip = [(i, drop[i]) for i in range(len(y)) if y_pred[i] == y[i] and y_pred_n[i] != y[i]]
    if wrong_flip:
        idxs = [i for i, _ in sorted(wrong_flip, key=lambda t: t[1], reverse=True)[:10]]
    else:
        idxs = list(np.argsort(drop)[::-1][:10])

    worst = []
    for i in idxs:
        worst.append(
            {
                "text": x[i],
                "noisy": x_noisy[i],
                "label": y[i],
                "pred_clean": y_pred[i],
                "pred_noisy": y_pred_n[i],
                "conf_clean": float(proba_c[i]),
                "conf_noisy": float(proba_n[i]),
                "conf_drop": float(drop[i]),
            }
        )

    return {
        "metrics": [
            {"split": "clean", "accuracy": acc, "macro_f1": f1},
            {"split": f"noisy({style})", "accuracy": acc_n, "macro_f1": f1_n},
        ],
        "worst_examples": worst,
    }


