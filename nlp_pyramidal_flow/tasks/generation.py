from __future__ import annotations

import re

from nlp_pyramidal_flow.utils.lang import detect_lang


def template_generate(text: str, prompt: str) -> str:
    """
    A lightweight template-based generator (no heavy model dependency).
    This is sufficient for a system demo, and can be replaced by LLM/transformers later.
    """
    lang = detect_lang(text + prompt, "自动")
    p = prompt.lower()

    if "总结" in prompt or "summary" in p:
        # naive summarization: take first sentence + key numbers/emails/urls
        first = re.split(r"[。！？!?]\s*", text.strip())[0].strip()
        extra = []
        extra += re.findall(r"[\w.+-]+@[\w-]+\.[\w.-]+", text)
        extra += re.findall(r"https?://\S+", text)
        extra += re.findall(r"\b\d+(?:[.,:/-]\d+)*\b", text)
        extra = list(dict.fromkeys(extra))[:3]
        if extra:
            return f"{first}（关键信息：{'，'.join(extra)}）"
        return first

    if "口语" in prompt or "colloquial" in p:
        if lang == "中文":
            return "说实话，" + text.replace("。", "！").replace("，", "，就") + "～"
        return "Honestly, " + text.replace(".", "!") + " :)"

    if "改写" in prompt or "paraphrase" in p:
        if lang == "中文":
            return text.replace("因为", "由于").replace("所以", "因此").replace("请", "麻烦")
        return text.replace("because", "since").replace("please", "kindly")

    if "生成" in prompt or "generate" in p:
        if lang == "中文":
            return f"基于输入内容，我建议：先确认关键信息，再给出行动项。\n- 关键信息：{text}\n- 行动项：请补充细节后再决定下一步。"
        return f"Based on the input, here are next steps:\n- Key info: {text}\n- Action: clarify requirements and proceed."

    return text


