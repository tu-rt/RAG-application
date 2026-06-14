from __future__ import annotations

import re
from typing import Iterable, Sequence


def normalize_answer(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"\s+", "", text)
    text = re.sub(r"[^\w\u4e00-\u9fff]", "", text)
    return text


def answer_in_text(answer: str, text: str) -> bool:
    a = normalize_answer(answer)
    t = normalize_answer(text)
    if not a:
        return False
    return a in t


def recall_at_k(
    references: Sequence[str],
    retrieved_texts: Sequence[str],
    k: int,
) -> bool:
    """任一参考答案出现在 Top-K 任一 chunk 中即命中。"""
    top = retrieved_texts[:k]
    for ref in references:
        for chunk in top:
            if answer_in_text(ref, chunk):
                return True
    return False


def f1_score(prediction: str, reference: str) -> float:
    pred_tokens = list(normalize_answer(prediction))
    ref_tokens = list(normalize_answer(reference))
    if not pred_tokens and not ref_tokens:
        return 1.0
    if not pred_tokens or not ref_tokens:
        return 0.0
    common = {}
    for t in pred_tokens:
        common[t] = common.get(t, 0) + 1
    match = 0
    ref_count = {}
    for t in ref_tokens:
        ref_count[t] = ref_count.get(t, 0) + 1
    for t, c in common.items():
        match += min(c, ref_count.get(t, 0))
    if match == 0:
        return 0.0
    precision = match / len(pred_tokens)
    recall = match / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def best_f1_over_refs(prediction: str, references: Iterable[str]) -> float:
    refs = [r for r in references if r]
    if not refs:
        return 0.0
    return max(f1_score(prediction, r) for r in refs)
