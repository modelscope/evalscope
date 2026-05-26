"""Rule-based scorer ported from the official GAIA leaderboard.

Source: https://huggingface.co/spaces/gaia-benchmark/leaderboard/blob/main/scorer.py
Copyright 2023 The GAIA Benchmark Authors, Apache License 2.0.
"""

import re
import string
from typing import Any, List, Tuple

from evalscope.utils.logger import get_logger

logger = get_logger()


def normalize_number_str(number_str: str) -> float:
    for char in ['$', '%', ',']:
        number_str = number_str.replace(char, '')
    try:
        return float(number_str)
    except ValueError:
        logger.warning(f'String {number_str} cannot be normalized to number str.')
        return float('inf')


def split_string(s: str, char_list: List[str] = [',', ';']) -> List[str]:
    pattern = f"[{''.join(char_list)}]"
    return re.split(pattern, s)


def normalize_str(input_str: str, remove_punct: bool = True) -> str:
    """Normalize a string by stripping whitespace, optionally punctuation, and lowercasing."""
    no_spaces = re.sub(r'\s', '', input_str)
    if remove_punct:
        translator = str.maketrans('', '', string.punctuation)
        return no_spaces.lower().translate(translator)
    return no_spaces.lower()


def question_scorer(model_answer: str, ground_truth: str) -> Tuple[bool, str]:
    """Compare model answer against ground truth using GAIA's official rules."""

    def is_float(element: Any) -> bool:
        try:
            float(element)
            return True
        except ValueError:
            return False

    if is_float(ground_truth):
        normalized_answer = normalize_number_str(model_answer)
        return (
            normalized_answer == float(ground_truth),
            f'Evaluated {model_answer} as a number.',
        )

    if any(char in ground_truth for char in [',', ';']):
        gt_elems = split_string(ground_truth)
        ma_elems = split_string(model_answer)

        if len(gt_elems) != len(ma_elems):
            return (
                False,
                f'Evaluated {model_answer} as a comma separated list, returned False because lists have different lengths.',
            )

        comparisons = []
        for ma_elem, gt_elem in zip(ma_elems, gt_elems):
            if is_float(gt_elem):
                normalized_ma_elem = normalize_number_str(ma_elem)
                comparisons.append(normalized_ma_elem == float(gt_elem))
            else:
                comparisons.append(
                    normalize_str(ma_elem, remove_punct=False) == normalize_str(gt_elem, remove_punct=False)
                )
        return all(comparisons), f'Evaluated {model_answer} as a comma separated list.'

    return (
        normalize_str(model_answer) == normalize_str(ground_truth),
        f'Evaluated {model_answer} as a string.',
    )


__all__ = ['question_scorer', 'normalize_number_str', 'normalize_str', 'split_string']
