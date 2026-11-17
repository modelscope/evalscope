# Copyright (c) Alibaba, Inc. and its affiliates.
"""Pure scoring functions for text-rendering (CVTG) benchmarks.

The implementations mirror the reference evaluators so scores stay comparable:
- CVTG-2K follows ``TextCrafter_Eval/unified_metrics_eval.py`` (NJU-PCALab/TextCrafter):
  ground-truth words are the single-quoted fragments of the prompt, Word Accuracy checks
  each ground-truth word for membership in the recognized words (repeated ground-truth
  words are not consumed), and NED is a word-level best-match Levenshtein distance.
- LongText-Bench follows ``textbench/summary_scores.py`` (X-Omni-Team/X-Omni): a Counter
  intersection between ground-truth and recognized tokens, word-level for English and
  character-level for Chinese. Chinese normalization removes all whitespace.

Counting functions return ``(score, numerator, denominator)`` pairs so callers can
aggregate at corpus level (total numerator / total denominator), matching the official
protocol instead of averaging per-image ratios.
"""

import re
from collections import Counter
from difflib import get_close_matches

# Characters kept by the LongText-Bench cleaner: CJK, latin letters, digits, whitespace
# and French accented letters (kept from the reference implementation).
_LTB_LEGAL_CHARS = re.compile(r'[一-龥a-zA-Z0-9\sàâäéèêëîïôöùûüçÀÂÄÉÈÊËÎÏÔÖÙÛÜÇ]')
_CJK = re.compile(r'[一-龥]')
_QUOTED = re.compile(r"'(.*?)'")


def detect_language(text: str) -> str:
    """LongText-Bench language for a sample: ``zh`` when the text contains CJK, else ``en``."""
    return 'zh' if _CJK.search(text or '') else 'en'


def extract_quoted_words(prompt: str) -> list:
    """Ground-truth words for CVTG-2K: lowercase words of every single-quoted fragment."""
    words = []
    for match in _QUOTED.findall(prompt or ''):
        words.extend(match.lower().split())
    return words


def _levenshtein(a: str, b: str) -> int:
    """Plain Levenshtein distance, avoiding a hard dependency on the editdistance wheel."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        current = [i]
        for j, cb in enumerate(b, 1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (ca != cb)))
        previous = current
    return previous[-1]


def word_accuracy_counts(gt_words: list, pred_words: list) -> tuple:
    """Official CVTG-2K Word Accuracy: every ground-truth word counts when it appears in
    the recognized words; repeated ground-truth words are not consumed.

    Returns ``(score, matched_words, total_gt_words)`` for corpus-level aggregation.
    """
    total = len(gt_words)
    if not total:
        return 0.0, 0, 0
    pred_set = set(pred_words)
    matched = sum(1 for word in gt_words if word in pred_set)
    return matched / total, matched, total


def word_accuracy(gt_words: list, pred_words: list) -> float:
    """Scalar convenience wrapper around :func:`word_accuracy_counts`."""
    return word_accuracy_counts(gt_words, pred_words)[0]


def word_ned_counts(gt_words: list, pred_words: list) -> tuple:
    """Official CVTG-2K NED: each ground-truth word matches its closest prediction
    (no consumption), similarity = 1 - distance / max(len) per word.

    Returns ``(score, similarity_sum, word_count)``; the official corpus aggregate is
    ``similarity_sum / word_count`` over all words of all samples.
    """
    total = len(gt_words)
    if not total:
        return 0.0, 0.0, 0
    similarities = []
    for gt_word in gt_words:
        closest = get_close_matches(gt_word, pred_words, n=1, cutoff=0)
        if not closest:
            similarities.append(0.0)
            continue
        edit_dist = _levenshtein(gt_word, closest[0])
        similarities.append(1 - edit_dist / (max(len(gt_word), len(closest[0])) + 1e-5))
    sim_sum = sum(similarities)
    return sim_sum / total, sim_sum, total


def word_level_ned(gt_words: list, pred_words: list) -> float:
    """Scalar convenience wrapper around :func:`word_ned_counts`."""
    return word_ned_counts(gt_words, pred_words)[0]


def normalize_recognized_text(text: str, lang: str) -> str:
    """Clean a string the way LongText-Bench does before matching.

    English: keep legal characters, collapse whitespace, lowercase. Chinese: keep legal
    characters and drop all whitespace (the official evaluator compares character
    sequences without any whitespace).
    """
    cleaned = ''.join(_LTB_LEGAL_CHARS.findall(text or ''))
    if lang == 'zh':
        return ''.join(cleaned.split())
    return ' '.join(cleaned.split()).lower()


def text_score_counts(gt_text: str, recognized: str, lang: str = None) -> tuple:
    """Official LongText-Bench Text Score: Counter-intersection match ratio.

    English compares word tokens, Chinese compares characters. ``lang=None`` detects the
    language from the ground-truth text (CJK -> ``zh``), so one run can cover both.

    Returns ``(score, matched_tokens, total_gt_tokens)`` for corpus-level aggregation.
    """
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
    gt_counter, pred_counter = Counter(gt_tokens), Counter(pred_tokens)
    matched = sum((gt_counter & pred_counter).values())
    return matched / total, matched, total


def text_score(gt_text: str, recognized: str, lang: str = None) -> float:
    """Scalar convenience wrapper around :func:`text_score_counts`."""
    return text_score_counts(gt_text, recognized, lang)[0]
