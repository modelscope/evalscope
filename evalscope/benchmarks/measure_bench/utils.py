# flake8: noqa: E501
"""Scoring utilities for MeasureBench, ported from the official evaluation code.

Reference: https://github.com/flageval-baai/MeasureBench/blob/main/evaluation/measure_bench_evaluator.py
"""
import math
import re
import unicodedata
from typing import Dict, List, Optional, Union


def normalize_string(text: str) -> str:
    """Replace special Unicode characters with their ASCII equivalents."""
    replace_dict = {'′': "'", '\u00a0': ' ', '‐': '-', '−': '-', '–': '-', '⋅': '·'}
    for k, v in replace_dict.items():
        text = text.replace(k, v)
    return text


def extract_numbers(text: str) -> List[float]:
    """Extract all numbers from *text* (integers, decimals, fractions).

    Fractions such as ``3/4`` are converted to floats.  Numbers appear in the
    order they are found in the text.
    """

    def norm_minus(s: str) -> str:
        return s.replace('\u2212', '-')

    pattern = re.compile(
        r"""
        (?P<fraction>[+\-\u2212]?\d+\s*[/]\s*[+\-\u2212]?\d+)
        |
        (?P<decimal>[+\-\u2212]?(?:\d*\.\d+|\d+\.\d*))
        |
        (?P<integer>[+\-\u2212]?\d+)
    """,
        re.VERBOSE,
    )

    out: List[float] = []
    for m in pattern.finditer(text):
        kind = m.lastgroup
        s = norm_minus(m.group(0)).strip()
        if kind == 'fraction':
            num_str, den_str = re.split(r'\s*/\s*', s, maxsplit=1)
            try:
                num, den = int(num_str), int(den_str)
                if den != 0:
                    out.append(num / den)
            except ValueError:
                continue
        else:
            try:
                out.append(float(s))
            except ValueError:
                continue
    return out


def extract_answer_text(prediction: str) -> str:
    """Extract the answer portion from the model's raw prediction.

    Looks for ``Answer:``/``Answer`` markers, ``\\boxed{...}`` patterns, or
    returns the full prediction if no marker is found.
    """
    indicators = ['Answer:', 'Answer', '答案：', '答案:', '答案']
    for indicator in indicators:
        if indicator in prediction:
            return prediction.split(indicator)[-1].strip()
    boxed_match = re.search(r'\\boxed\{([^}]+)\}', prediction)
    if boxed_match:
        return boxed_match.group(1).strip()
    return prediction


def time_to_seconds(parts: List[str]) -> int:
    """Convert a HH:MM:SS or MM:SS time tuple to total seconds."""
    scale = [3600, 60, 1]
    seconds = 0
    for i, part in enumerate(parts):
        if part != '':
            seconds += scale[i] * int(part)
    return seconds


def _interval_matching(
    answer_text: str,
    interval: List[Union[float, str]],
    units: List[str],
) -> Dict:
    """Check whether *answer_text* falls within *interval* and contains a valid unit.

    Returns a result dict with keys ``all_correct``, ``number_correct``,
    ``number_error_rate``, and ``unit_correct``.
    """
    eval_result: Dict = {
        'all_correct': 0,
        'number_correct': 0,
        'number_error_rate': None,
        'unit_correct': 0,
    }
    pred_lower = unicodedata.normalize('NFKC', answer_text.lower()).replace('µ', 'μ')

    for unit in units:
        unit_lower = unicodedata.normalize('NFKC', unit.lower()).replace('µ', 'μ')
        if unit_lower in pred_lower:
            eval_result['unit_correct'] = 1
    if len(units) == 0:
        eval_result['unit_correct'] = None

    # Time interval — values are strings like "11:15:59"
    if isinstance(interval[0], str):
        left_seconds = time_to_seconds(interval[0].split(':'))
        right_seconds = time_to_seconds(interval[1].split(':'))
        time_pattern = r'\b(\d{1,2}):(\d{2})(?::(\d{2}))?\b'
        matches = re.findall(time_pattern, answer_text)
        if not matches:
            return eval_result
        match = matches[0]
        # Pad to 3 elements [H, M, S]
        parts = list(match)
        while len(parts) < 3:
            parts.append('')
        pred_ans: Union[int, float] = time_to_seconds(parts)
        left_interval: Union[int, float] = left_seconds
        right_interval: Union[int, float] = right_seconds
    else:
        # Numeric interval
        left_interval = float(interval[0])
        right_interval = float(interval[1])
        numbers = extract_numbers(answer_text)
        if not numbers or math.isinf(numbers[-1]) or math.isnan(numbers[-1]):
            return eval_result
        pred_ans = numbers[-1]

    if pred_ans < left_interval or pred_ans > right_interval:
        eps = 1e-6
        eval_result['number_error_rate'] = min(
            abs((pred_ans - left_interval) / (left_interval + eps)),
            abs((pred_ans - right_interval) / (right_interval + eps)),
        )
        return eval_result

    eval_result['number_correct'] = 1
    eval_result['number_error_rate'] = 0.0
    if eval_result['unit_correct'] is None or eval_result['unit_correct'] == 1:
        eval_result['all_correct'] = 1
    return eval_result


def _is_current_better(result: Dict, best_result: Optional[Dict]) -> bool:
    """Return True if *result* is better than *best_result*."""
    if best_result is None:
        return True
    cur_unit = result['unit_correct'] if result['unit_correct'] is not None else 1
    best_unit = best_result['unit_correct'] if best_result['unit_correct'] is not None else 1
    if result['number_correct'] + cur_unit > best_result['number_correct'] + best_unit:
        return True
    if result['number_error_rate'] is not None and best_result['number_error_rate'] is not None:
        if result['number_error_rate'] < best_result['number_error_rate']:
            return True
    return False


def interval_matching(
    answer_text: str,
    interval: List[Union[float, str]],
    units: List[str],
) -> Dict:
    """Evaluate a single interval match."""
    return _interval_matching(answer_text, interval, units)


def multi_interval_matching(
    answer_text: str,
    intervals: List[List[Union[float, str]]],
    units: List[Union[str, List[str]]],
) -> Dict:
    """Evaluate against multiple valid intervals and return the best match.

    ``units`` may be a flat list (shared across intervals) or a list-of-lists
    (one list per interval), mirroring the official evaluator format.
    """
    # Normalise units: must be a list of lists, one per interval
    if len(units) == 0:
        per_interval_units: List[List[str]] = [[] for _ in intervals]
    elif isinstance(units[0], list):
        per_interval_units = units  # type: ignore[assignment]
    else:
        # Flat list — replicate for each interval
        per_interval_units = [list(units)] * len(intervals)  # type: ignore[list-item]

    best_result: Optional[Dict] = None
    for interval, unit in zip(intervals, per_interval_units):
        result = _interval_matching(answer_text, interval, unit)
        if result['all_correct'] == 1:
            return result
        if _is_current_better(result, best_result):
            best_result = result

    return best_result or {
        'all_correct': 0,
        'number_correct': 0,
        'number_error_rate': None,
        'unit_correct': 0,
    }
