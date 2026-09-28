"""Compatibility entry points backed exclusively by isolated Math-Verify calls."""

import re
import warnings
from typing import Any, Literal

from .contracts import InvalidMathReference, MathRequest, MathResult
from .runtime import execute_math


def compare_answers(
    prediction: str,
    reference: str,
    *,
    prediction_mode: Literal['output', 'fragment'] = 'fragment',
    absolute_tolerance: list[float] | None = None,
    relative_tolerance: float | None = None,
    multiple_answers: bool = False,
    integer_only: bool = False,
    numeric_reference: bool = False,
    validate_reference: bool = True,
) -> MathResult:
    """Compare parsed mathematics with optional benchmark numeric policy."""
    result = execute_math(
        MathRequest(
            prediction=prediction,
            reference=reference,
            prediction_mode=prediction_mode,
            absolute_tolerance=absolute_tolerance,
            relative_tolerance=relative_tolerance,
            multiple_answers=multiple_answers,
            integer_only=integer_only,
            numeric_reference=numeric_reference,
        )
    )
    if validate_reference and not result.reference_valid:
        raise InvalidMathReference(f'Invalid mathematical reference ({result.reason}): {reference!r}')
    return result


def extract_answer(pred_str: str, use_last_number: bool = True) -> str:
    """Extract display text from full output, preserving percent signs.

    ``use_last_number`` is deprecated: extraction follows Math-Verify's strategy.
    Unparsed fallback text is display-only and cannot establish mathematical equality.
    """
    if not use_last_number:
        warnings.warn(
            'use_last_number is deprecated; Math-Verify controls extraction', DeprecationWarning, stacklevel=2
        )
    return execute_math(MathRequest(operation='extract', prediction=pred_str, prediction_mode='output')).extracted


def strip_answer_string(string: str) -> str:
    """Return upstream-normalized display text for an already extracted math fragment."""
    return execute_math(MathRequest(operation='extract', prediction=string, prediction_mode='fragment')).extracted


def math_equal(
    prediction: Any,
    reference: Any,
    include_percentage: bool = True,
    is_close: bool = True,
    timeout: bool = False,
) -> bool:
    """Compare using HF defaults and a mandatory parent deadline.

    Legacy algorithm switches are deprecated and no longer alter scoring rules.
    ``10`` and ``10%`` can match under HF's integer percentage compatibility.
    """
    if not include_percentage or not is_close or timeout:
        warnings.warn(
            'include_percentage, is_close and timeout are deprecated; Math-Verify uses HF defaults '
            'and an enforced 15-second deadline',
            DeprecationWarning,
            stacklevel=2,
        )
    return compare_answers(
        '' if prediction is None else str(prediction),
        '' if reference is None else str(reference),
        validate_reference=False,
    ).matched


def extract_boxed_answers(text: str) -> list[str]:
    """Delegate boxed boundaries upstream while retaining ordered subquestion payloads."""
    return execute_math(MathRequest(operation='boxed', prediction=text)).parts


def parse_digits(num: Any) -> float | None:
    """Return an upstream-parsed real number, without a separate numeric parser."""
    return execute_math(MathRequest(operation='number', prediction=str(num))).number


def is_digit(num: Any) -> bool:
    """Report whether Math-Verify parsed a finite real number."""
    return parse_digits(num) is not None


def numeric_equal(prediction: float, reference: float) -> bool:
    """Delegate the legacy numeric helper to the common HF comparison."""
    return math_equal(prediction, reference)


def symbolic_equal(a: Any, b: Any) -> bool:
    """Delegate the legacy symbolic helper to the common HF comparison."""
    return math_equal(a, b)


def convert_word_number(text: str) -> str:
    """Deprecated word-number conversion; supported numbers are parsed upstream."""
    warnings.warn(
        'convert_word_number is deprecated; only upstream-supported notation is parsed',
        DeprecationWarning,
        stacklevel=2,
    )
    result = execute_math(MathRequest(operation='number', prediction=text))
    return result.extracted if result.number is not None else text


def str_to_pmatrix(input_str: str) -> str:
    """Deprecated matrix rewrite; render the upstream-parsed mathematics instead."""
    warnings.warn(
        'str_to_pmatrix is deprecated; matrix notation is handled by Math-Verify', DeprecationWarning, stacklevel=2
    )
    return execute_math(MathRequest(operation='render', prediction=input_str)).extracted


def choice_answer_clean(pred: str) -> str:
    """Retain the legacy categorical text cleanup independently of math scoring."""
    pred = pred.strip('\n').rstrip('.').rstrip('/').strip(' ').lstrip(':')
    matches = re.findall(r'\b(A|B|C|D|E)\b', pred.upper())
    return (matches[-1] if matches else pred.strip().strip('.')).rstrip('.').rstrip('/')
