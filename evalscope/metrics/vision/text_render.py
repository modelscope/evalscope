# Copyright (c) Alibaba, Inc. and its affiliates.
"""Official TextCrafter and X-Omni scoring rules, returning (score, numerator, denominator)."""

import re
from collections import Counter
from difflib import get_close_matches

from evalscope.metrics.utils import levenshtein_distance

# Preserve the official cleaner's French accented characters.
_LTB_LEGAL_CHARS = re.compile(r'[一-龥a-zA-Z0-9\sàâäéèêëîïôöùûüçÀÂÄÉÈÊËÎÏÔÖÙÛÜÇ]')
_CJK = re.compile(r'[一-龥]')
_QUOTED = re.compile(r"'(.*?)'")


def detect_language(text: str) -> str:
    """LongText-Bench language for a sample: ``zh`` when the text contains CJK, else ``en``."""
    return 'zh' if _CJK.search(text or '') else 'en'


def extract_quoted_words(prompt: str) -> list:
    """Ground-truth words for CVTG-2K: lowercase words of every single-quoted fragment."""
    return [word for fragment in _QUOTED.findall(prompt or '') for word in fragment.lower().split()]


def word_accuracy_counts(gt_words: list, pred_words: list) -> tuple:
    """Count membership matches; repeated ground-truth words are not consumed."""
    total = len(gt_words)
    if not total:
        return 0.0, 0, 0
    pred_set = set(pred_words)
    matched = sum(1 for word in gt_words if word in pred_set)
    return matched / total, matched, total


def word_ned_counts(gt_words: list, pred_words: list) -> tuple:
    """Sum best-match word similarities using the official difflib selection and epsilon."""
    total = len(gt_words)
    if not total:
        return 0.0, 0.0, 0
    # The official OCR evaluator substitutes one empty word when recognition finds no text.
    pred_words = pred_words or ['']
    sim_sum = 0.0
    for gt_word in gt_words:
        closest = get_close_matches(gt_word, pred_words, n=1, cutoff=0)[0]
        sim_sum += 1 - levenshtein_distance(gt_word, closest) / (max(len(gt_word), len(closest)) + 1e-5)
    return sim_sum / total, sim_sum, total


def normalize_recognized_text(text: str, lang: str) -> str:
    """Keep legal characters; remove Chinese whitespace or lowercase English words."""
    cleaned = ''.join(_LTB_LEGAL_CHARS.findall(text or ''))
    if lang == 'zh':
        return ''.join(cleaned.split())
    return ' '.join(cleaned.split()).lower()


def text_score_counts(gt_text: str, recognized: str, lang: str = None) -> tuple:
    """Count matched English words or Chinese characters, detecting language when omitted."""
    lang = lang or detect_language(gt_text or '')
    gt_text = normalize_recognized_text(gt_text, lang)
    recognized = normalize_recognized_text(recognized, lang)
    if lang == 'zh':
        gt_tokens, pred_tokens = list(gt_text), list(recognized)
    else:
        gt_tokens, pred_tokens = gt_text.split(), recognized.split()
    total = len(gt_tokens)
    if not total:
        return 0.0, 0, 0
    matched = sum((Counter(gt_tokens) & Counter(pred_tokens)).values())
    return matched / total, matched, total
