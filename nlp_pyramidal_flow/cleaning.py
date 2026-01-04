from __future__ import annotations

import re


_FULLWIDTH_TO_HALFWIDTH = str.maketrans(
    {
        "／": "/",
        "：": ":",
        "，": ",",
        "。": ".",
        "！": "!",
        "？": "?",
        "（": "(",
        "）": ")",
        "【": "[",
        "】": "]",
    }
)


def detect_issues(text: str) -> list[dict]:
    issues: list[dict] = []
    if re.search(r"\s{2,}", text):
        issues.append({"issue": "多余空白", "detail": "存在连续空格/制表符"})
    if re.search(r"[。．\.]{2,}|[!！]{2,}|[?？]{2,}|[，,]{2,}", text):
        issues.append({"issue": "标点重复", "detail": "存在重复标点（如 !!!、。。。）"})
    if re.search(r"[／：，。！？（）【】]", text):
        issues.append({"issue": "全角标点", "detail": "检测到全角标点（可统一为半角）"})
    if re.search(r"\w+\s+@\s+\w+\.\w+", text):
        issues.append({"issue": "邮箱被空白打断", "detail": "类似 zhangsan @example.com"})
    if re.search(r"([\u4e00-\u9fff])\s+([\u4e00-\u9fff])", text):
        issues.append({"issue": "中文被异常空白打断", "detail": "中文字符之间夹了空格"})
    if re.search(r"(.)\1\1+", text):
        issues.append({"issue": "字符重复", "detail": "存在 3 次及以上的同字符重复"})
    if not issues:
        issues.append({"issue": "未发现明显问题", "detail": "文本看起来较干净"})
    return issues


def clean_text(text: str) -> str:
    x = text
    x = x.translate(_FULLWIDTH_TO_HALFWIDTH)

    # fix spaced emails: "a @ b.com" -> "a@b.com"
    x = re.sub(r"(\w)\s+@\s+(\w)", r"\1@\2", x)

    # collapse repeated punctuation
    x = re.sub(r"[。．\.]{2,}", "。", x)
    x = re.sub(r"[!！]{2,}", "！", x)
    x = re.sub(r"[?？]{2,}", "？", x)
    x = re.sub(r"[，,]{2,}", "，", x)

    # remove whitespace between Chinese chars
    x = re.sub(r"([\u4e00-\u9fff])\s+([\u4e00-\u9fff])", r"\1\2", x)

    # normalize multi-space
    x = re.sub(r"\s{2,}", " ", x).strip()

    # reduce excessive repeats: "差差差差" -> "差差"
    x = re.sub(r"(.)\1{2,}", r"\1\1", x)
    return x


