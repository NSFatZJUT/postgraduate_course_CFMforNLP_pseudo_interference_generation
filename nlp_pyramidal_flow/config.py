from dataclasses import dataclass


@dataclass(frozen=True)
class PIGConfig:
    """
    Pseudo-Interference Generation (PIG) config for NLP.

    Pyramidal levels (coarse-to-fine):
    - sentence: discourse / connectives / fillers / light reordering
    - word: synonym-ish replacement / insertions
    - subword: whitespace, punctuation, fullwidth/halfwidth, tokenization sensitivity
    - char: typos, homoglyphs, homophones, keyboard noise
    """

    seed: int = 42
    lang_hint: str = "自动"  # 自动/中文/英文

    intensity_sentence: float = 0.25
    intensity_word: float = 0.25
    intensity_subword: float = 0.20
    intensity_char: float = 0.20

    max_ops: int = 16
    protect_numbers: bool = True


