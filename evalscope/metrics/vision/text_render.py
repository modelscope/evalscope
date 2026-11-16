# Copyright (c) Alibaba, Inc. and its affiliates.
"""Pure scoring functions for text-rendering (CVTG) benchmarks.

The implementations mirror the reference evaluators so scores stay comparable:
- CVTG-2K follows ``TextCrafter_Eval/unified_metrics_eval.py`` (NJU-PCALab/TextCrafter):
  ground-truth words are the single-quoted fragments of the prompt, Word Accuracy is an
  exact word match and NED is a word-level best-match Levenshtein distance.
- LongText-Bench follows ``textbench/summary_scores.py`` (X-Omni-Team/X-Omni): a Counter
  intersection between ground-truth and recognized tokens, word-level for English and
  character-level for Chinese, normalized per sample.
"""

import re
from collections import Counter
from difflib import get_close_matches

# Characters kept by the LongText-Bench cleaner: CJK, latin letters, digits, whitespace
# and French accented letters (kept from the reference implementation).
_LTB_LEGAL_CHARS = re.compile(r'[一-龥a-zA-Z0-9\sàâäéèêëîïôöùûüçÀÂÄÉÈÊËÎÏÔÖÙÛÜÇ]')
_CJK = re.compile(r'[一-龥]')
_QUOTED = re.compile(r"'(.*?)'")


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


def word_accuracy(gt_words: list, pred_words: list) -> float:
    """Exact word match ratio over the ground-truth word multiset (CVTG-2K)."""
    if not gt_words:
        return 0.0
    pred_counter = Counter(pred_words)
    matched = sum(min(count, pred_counter.get(word, 0)) for word, count in Counter(gt_words).items())
    return matched / len(gt_words)


def word_level_ned(gt_words: list, pred_words: list) -> float:
    """Word-level NED: each ground-truth word matches its closest prediction (CVTG-2K)."""
    if not gt_words:
        return 0.0
    distances = []
    for gt_word in gt_words:
        closest = get_close_matches(gt_word, pred_words, n=1, cutoff=0)
        if not closest:
            distances.append(0.0)
            continue
        edit_dist = _levenshtein(gt_word, closest[0])
        distances.append(1 - edit_dist / (max(len(gt_word), len(closest[0])) + 1e-5))
    return sum(distances) / len(distances)


def normalize_recognized_text(text: str, lang: str) -> str:
    """Clean a recognized string the way LongText-Bench does before matching."""
    cleaned = ''.join(_LTB_LEGAL_CHARS.findall(text or ''))
    if lang == 'zh':
        return cleaned
    return ' '.join(cleaned.split()).lower()


def text_score(gt_text: str, recognized: str, lang: str) -> float:
    """LongText-Bench Text Score: Counter-intersection match ratio.

    English compares word tokens, Chinese compares characters; the ratio is per sample.
    """
    gt_text = normalize_recognized_text(gt_text, lang)
    recognized = normalize_recognized_text(recognized, lang)
    if not gt_text:
        return 0.0
    if lang == 'zh':
        gt_tokens, pred_tokens = list(gt_text), list(recognized)
    else:
        gt_tokens, pred_tokens = gt_text.split(), recognized.split()
    if not gt_tokens:
        return 0.0
    gt_counter, pred_counter = Counter(gt_tokens), Counter(pred_tokens)
    matched = sum((gt_counter & pred_counter).values())
    return matched / sum(gt_counter.values())
