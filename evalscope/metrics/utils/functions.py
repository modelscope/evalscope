# Copyright (c) Alibaba, Inc. and its affiliates.
# Copyright (c) EleutherAI. and its affiliates.
# Copyright (c) OpenAI. and its affiliates.

import editdistance
import itertools
import math
import numpy as np
from typing import Dict, List, Union


def normalize_text(text: str) -> str:
    """Normalize text by lowering case and stripping whitespace."""
    return text.strip().lower()


def mean(arr: list):
    if not arr:
        return 0.0

    if isinstance(arr[0], list):
        arr = [item for sublist in arr for item in sublist]
    return sum(arr) / len(arr)


def simple_f1_score(scores: tuple) -> float:
    score1 = scores[0]
    score2 = scores[1]
    score1 = np.mean(score1) if len(score1) > 0 else 0.0
    score2 = np.mean(score2) if len(score2) > 0 else 0.0

    if score1 == 0 and score2 == 0:
        return 0.0
    else:
        return 2 * score1 * score2 / (score1 + score2)


def micro_mean(items):
    try:
        return sum([item.score * item.num for item in items]) / sum([item.num for item in items])
    except ZeroDivisionError:
        return 0.0


def macro_mean(items):
    try:
        return sum([item.score for item in items]) / len(items)
    except ZeroDivisionError:
        return 0.0


def bleu_ngram_one_sample(predict: str, reference: str):
    """
    Calculate BLEU-1, BLEU-2, BLEU-3, and BLEU-4 scores

    Args:
        predict: predicted text
        reference: reference text

    Returns:
        {
            'bleu-1': 0.8,
            'bleu-2': 0.45,
            'bleu-3': 0.0,
            'bleu-4': 0.0
        }

    """
    import jieba
    from nltk import word_tokenize
    from nltk.translate.bleu_score import sentence_bleu

    def is_contains_chinese(strs):
        for _char in strs:
            if '\u4e00' <= _char <= '\u9fa5':
                return True
        return False

    predict = list(jieba.cut(predict)) if is_contains_chinese(predict) else word_tokenize(predict)
    reference = [list(jieba.cut(reference))] if is_contains_chinese(reference) else [word_tokenize(reference)]

    result = dict()
    result['bleu-1'] = sentence_bleu(reference, predict, weights=(1, 0, 0, 0))
    result['bleu-2'] = sentence_bleu(reference, predict, weights=(0, 1, 0, 0))
    result['bleu-3'] = sentence_bleu(reference, predict, weights=(0, 0, 1, 0))
    result['bleu-4'] = sentence_bleu(reference, predict, weights=(0, 0, 0, 1))

    return result


def exact_match(gold: str, pred: str) -> float:
    if not pred:
        return 0

    return 1 if gold.strip() == pred.strip() else 0


def calculate_pass_at_k(
    num_samples: Union[int, List[int], np.ndarray],
    num_correct: Union[List[int], np.ndarray],
    k: int = 1
) -> np.ndarray:
    """
    Estimates pass@k of each problem and returns them in an array.
    Examples:
        >>> import numpy as np
        >>> from typing import Union
        >>> total = np.array([5, 5, 5])
        >>> correct = np.array([2, 4, 2])
        >>> calculate_pass_at_k(total, correct, 1)
        result: "array([0.4, 0.8, 0.4])"
    """

    def estimator(n: int, c: int, k: int) -> float:
        """
        Calculates 1 - comb(n - c, k) / comb(n, k).
        """
        if n - c < k:
            return 1.0
        return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))

    if isinstance(num_samples, int):
        num_samples_it = itertools.repeat(num_samples, len(num_correct))
    else:
        assert len(num_samples) == len(num_correct)
        num_samples_it = iter(num_samples)

    return np.array([estimator(int(n), int(c), k) for n, c in zip(num_samples_it, num_correct)])


def calculate_pass_hat_k(num_trials: int, success_count: int, k: int) -> float:
    """
    Compute the pass^k metric for the given number of trials, success count, and k.
    from https://arxiv.org/pdf/2406.12045
    Args:
        num_trials: The number of trials.
        success_count: The number of successful trials.
        k: The number of trials to consider.
    Returns:
        The pass^k metric.
    """
    if num_trials < k:
        raise ValueError(f'Number of trials {num_trials} is less than k {k}.')
    return math.comb(success_count, k) / math.comb(num_trials, k)


def levenshtein_distance(s1: Union[str, List[str]], s2: Union[str, List[str]]) -> int:
    """Compute Levenshtein distance using the editdistance library.

    Supports both strings and token sequences (lists).
    """
    return editdistance.eval(s1, s2)
