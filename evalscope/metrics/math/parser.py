"""Mathematical answer extraction and comparison delegated to Math-Verify."""

import math
import re
import warnings
from decimal import Decimal, InvalidOperation, localcontext
from functools import wraps
from typing import Any, Callable, Literal, ParamSpec, TypeVar

from .contracts import InvalidMathReference, MathEvaluationError, MathResult

_P = ParamSpec('_P')
_R = TypeVar('_R')


def _math_call(function: Callable[_P, _R]) -> Callable[_P, _R]:
    @wraps(function)
    def wrapped(*args: _P.args, **kwargs: _P.kwargs) -> _R:
        try:
            return function(*args, **kwargs)
        except MathEvaluationError:
            raise
        except Exception as exc:
            raise MathEvaluationError(f'{type(exc).__name__}: {exc}') from exc

    return wrapped


def _parse(text: str, mode: str) -> tuple[list[Any], str]:
    from math_verify import ExprExtractionConfig, LatexExtractionConfig, parse

    text = text.strip()
    if not text:
        return [], ''
    original = text
    scientific_literal = re.fullmatch(r'[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)[eE][+-]?[0-9]+', text)
    if scientific_literal:
        # The pinned LaTeX parser recognizes uppercase E notation; lowercase e is Euler's constant.
        # Canonicalize complete numeric literals without evaluating them locally.
        text = f'${text.replace("e", "E")}$'
    if mode == 'fragment' and not text.startswith(('$', r'\(', r'\[', r'\boxed', r'\fbox')):
        text = f'${text}$'
    result = parse(
        text,
        extraction_config=[LatexExtractionConfig(boxed_match_priority=0), ExprExtractionConfig()],
        fallback_mode='first_match',
        extraction_mode='first_match',
        parsing_timeout=None,
    )
    objects = [item for item in result if not isinstance(item, str)]
    display = next((item for item in result if isinstance(item, str)), '')
    if objects and re.fullmatch(r'[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)e[+-]?[0-9]+', display):
        canonical = parse(
            f'${display.replace("e", "E")}$',
            extraction_config=[LatexExtractionConfig()],
            fallback_mode='first_match',
            extraction_mode='first_match',
            parsing_timeout=None,
        )
        objects = [item for item in canonical if not isinstance(item, str)]
    if scientific_literal:
        display = original
    if len(objects) == 1 and '%' not in display:
        from math_verify.grader import get_pct_val

        # Expr extraction's fallback omits the percent suffix. Restore display only;
        # the upstream object and HF comparison semantics stay untouched.
        if get_pct_val(objects[0]) is not None:
            display += '%'
    return objects, display


def _boxed_parts(text: str) -> tuple[list[str], str]:
    from latex2sympy2_extended.math_normalization import extract_boxed_content

    # Preserve complete subquestion boundaries and order before upstream normalization.
    # An unfinished outer answer must not promote an inner box to a new subquestion.
    parts = []
    extracted = ''
    end = 0
    for match in re.finditer(r'\\(?:boxed|fbox)\s*\{', text):
        if match.start() < end:
            continue
        depth = 1
        for index in range(match.end(), len(text)):
            if text[index] == '{':
                depth += 1
            elif text[index] == '}':
                depth -= 1
            if depth == 0:
                stop = index + 1
                parts.append(extract_boxed_content(text[match.start() : stop], mode='all'))
                # Retain the final answer's same-line suffix for instrument units.
                extracted = text[match.end() : stop - 1] + text[stop:].split('\n', 1)[0]
                end = stop
                break
        else:
            extracted = ''
            break
    return parts, extracted


def _number(objects: list[Any]) -> float | None:
    if len(objects) != 1 or not getattr(objects[0], 'is_number', False) or not objects[0].is_real:
        return None
    value = float(objects[0])
    return value if math.isfinite(value) else None


def _verify(gold: Any, prediction: Any) -> bool:
    from math_verify import verify

    return verify(
        [gold],
        [prediction],
        strict=True,
        float_rounding=6,
        numeric_precision=15,
        allow_set_relation_comp=False,
        timeout_seconds=None,
        raise_on_error=True,
    )


def _numeric_close(gold: Any, prediction: Any, absolute: float, relative: float) -> bool:
    # Only explicit benchmark numeric tolerance is local policy.
    def decimal_value(value: Any) -> Decimal:
        try:
            return Decimal(str(value))
        except InvalidOperation:
            return Decimal(str(value.evalf(15)))

    # Decimal thresholds retain exact dataset boundaries such as 1.01 vs 1 +/- 0.01.
    with localcontext() as context:
        context.prec = 50
        pred, ref = decimal_value(prediction), decimal_value(gold)
        threshold = max(Decimal(str(absolute)), Decimal(str(relative)) * max(abs(pred), abs(ref)))
        return abs(pred - ref) <= threshold


def _components(objects: list[Any]) -> list[Any]:
    from sympy import FiniteSet, Tuple

    if len(objects) == 1 and isinstance(objects[0], (FiniteSet, Tuple)):
        # The pinned upstream preserves input order for per-component tolerances.
        return list(getattr(objects[0], '_unsorted_args', objects[0].args))
    return objects


def _answer_components(objects: list[Any], source: str) -> list[Any]:
    # This is answer layout, not a mathematical parser. Parse each declared component
    # upstream before set conversion can erase duplicates (e.g. two answers both 1).
    from latex2sympy2_extended.math_normalization import NormalizationConfig, normalize_latex

    display = normalize_latex(source, NormalizationConfig(boxed='last'))
    depth = 0
    start = 0
    fragments = []
    for index, char in enumerate(display):
        if char in '([{':
            depth += 1
        elif char in ')]}':
            depth -= 1
        elif char == ',' and depth == 0 and (index == 0 or display[index - 1] != '\\'):
            fragments.append(display[start:index])
            start = index + 1
    if not fragments:
        return _components(objects)
    fragments.append(display[start:])
    parts = [_parse(fragment, 'fragment')[0] for fragment in fragments]
    return [part[0] for part in parts] if all(len(part) == 1 for part in parts) else []


def _integer_literal(text: str) -> bool:
    try:
        value = Decimal(text.replace(',', '').strip())
        return value.is_finite() and value == value.to_integral_value()
    except InvalidOperation:
        return False


@_math_call
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
    if prediction_mode not in ('output', 'fragment'):
        raise ValueError('Prediction mode must be output or fragment')
    if absolute_tolerance is not None:
        absolute_tolerance = [float(value) for value in absolute_tolerance]
        if not absolute_tolerance or any(value < 0 or not math.isfinite(value) for value in absolute_tolerance):
            raise ValueError('Absolute tolerances must be finite and nonnegative')
    if relative_tolerance is not None:
        relative_tolerance = float(relative_tolerance)
        if relative_tolerance < 0 or not math.isfinite(relative_tolerance):
            raise ValueError('Relative tolerance must be finite and nonnegative')
    predictions, display = _parse(prediction, prediction_mode)
    result = MathResult(extracted=display, prediction_valid=bool(predictions))

    def finish() -> MathResult:
        if validate_reference and not result.reference_valid:
            raise InvalidMathReference(f'Invalid mathematical reference ({result.reason}): {reference!r}')
        return result

    golds, gold_display = _parse(reference, 'fragment')
    result.reference_valid = bool(golds)
    if not golds:
        result.reason = 'invalid_reference'
        return finish()
    if numeric_reference and (
        len(golds) != 1 or not golds[0].is_number or not golds[0].is_real or not golds[0].is_finite
    ):
        result.reference_valid = False
        result.reason = 'invalid_numeric_reference'
        return finish()
    if integer_only and not _integer_literal(gold_display):
        result.reference_valid = False
        result.reason = 'invalid_integer_reference'
        return finish()
    if not predictions:
        result.reason = 'prediction_not_parsed'
        return finish()
    if integer_only and not _integer_literal(display):
        result.reason = 'integer_literal_required'
        return finish()
    if multiple_answers:
        golds = _answer_components(golds, reference)
        predictions = _answer_components(predictions, prediction)
        if not golds:
            result.reference_valid = False
            result.reason = 'invalid_reference'
            return finish()
        if len(golds) != len(predictions):
            result.reason = 'answer_count_mismatch'
            return finish()
    absolute = absolute_tolerance
    if absolute is not None and len(absolute) not in (1, len(golds)):
        raise ValueError('Tolerance count must be one or equal to the reference answer count')

    def matches(gold: Any, prediction: Any, index: int) -> bool:
        if absolute is not None or relative_tolerance is not None:
            from sympy import Equality

            # A declared numeric component may include its variable label (Delta=...).
            # Delegate label equivalence, then apply the component's explicit tolerance.
            if (
                isinstance(gold, Equality)
                and isinstance(prediction, Equality)
                and gold.rhs.is_number
                and prediction.rhs.is_number
                and _verify(gold.lhs, prediction.lhs)
            ):
                gold, prediction = gold.rhs, prediction.rhs
            if (
                getattr(gold, 'is_number', False)
                and getattr(prediction, 'is_number', False)
                and gold.is_real
                and prediction.is_real
            ):
                return _numeric_close(
                    gold,
                    prediction,
                    absolute[min(index, len(absolute) - 1)] if absolute else 0,
                    relative_tolerance or 0,
                )
        return _verify(gold, prediction)

    if multiple_answers:

        def assign(index: int, remaining: list[Any]) -> bool:
            if index == len(golds):
                return True
            return any(
                matches(golds[index], pred, index) and assign(index + 1, remaining[:i] + remaining[i + 1 :])
                for i, pred in enumerate(remaining)
            )

        result.matched = assign(0, predictions)
    else:
        result.matched = any(matches(gold, pred, i) for i, gold in enumerate(golds) for pred in predictions)
    result.reason = 'matched' if result.matched else 'not_equivalent'
    return finish()


@_math_call
def extract_answer(pred_str: str, use_last_number: bool = True) -> str:
    """Extract display text from full output, preserving percent signs.

    ``use_last_number`` is deprecated: extraction follows Math-Verify's strategy.
    Unparsed fallback text is display-only and cannot establish mathematical equality.
    """
    if not use_last_number:
        warnings.warn(
            'use_last_number is deprecated; Math-Verify controls extraction', DeprecationWarning, stacklevel=2
        )
    return _parse(pred_str, 'output')[1]


@_math_call
def strip_answer_string(string: str) -> str:
    """Return upstream-normalized display text for an already extracted math fragment."""
    return _parse(string, 'fragment')[1]


def math_equal(
    prediction: Any,
    reference: Any,
    include_percentage: bool = True,
    is_close: bool = True,
    timeout: bool = False,
) -> bool:
    """Compare using HF defaults with library signal timers disabled.

    Legacy algorithm switches are deprecated and no longer alter scoring rules.
    ``10`` and ``10%`` can match under HF's integer percentage compatibility.
    """
    if not include_percentage or not is_close or timeout:
        warnings.warn(
            'include_percentage, is_close and timeout are deprecated and do not change Math-Verify scoring',
            DeprecationWarning,
            stacklevel=2,
        )
    return compare_answers(
        '' if prediction is None else str(prediction),
        '' if reference is None else str(reference),
        validate_reference=False,
    ).matched


@_math_call
def extract_boxed_answers(text: str) -> list[str]:
    """Delegate boxed boundaries upstream while retaining ordered subquestion payloads."""
    return _boxed_parts(text)[0]


@_math_call
def extract_boxed_answer_text(text: str) -> str:
    """Return the final complete boxed payload and its same-line suffix, including units.

    An unfinished final box returns an empty answer instead of exposing its digits.
    """
    return _boxed_parts(text)[1]


@_math_call
def parse_digits(num: Any, *, prediction_mode: Literal['output', 'fragment'] = 'fragment') -> float | None:
    """Return an upstream-parsed real number, without a separate numeric parser."""
    return _number(_parse(str(num), prediction_mode)[0])


def is_digit(num: Any) -> bool:
    """Report whether Math-Verify parsed a finite real number."""
    return parse_digits(num) is not None


def numeric_equal(prediction: float, reference: float) -> bool:
    """Delegate the legacy numeric helper to the common HF comparison."""
    return math_equal(prediction, reference)


def symbolic_equal(a: Any, b: Any) -> bool:
    """Delegate the legacy symbolic helper to the common HF comparison."""
    return math_equal(a, b)


@_math_call
def convert_word_number(text: str) -> str:
    """Deprecated word-number conversion; supported numbers are parsed upstream."""
    warnings.warn(
        'convert_word_number is deprecated; only upstream-supported notation is parsed',
        DeprecationWarning,
        stacklevel=2,
    )
    objects, display = _parse(text, 'fragment')
    return display if _number(objects) is not None else text


@_math_call
def str_to_pmatrix(input_str: str) -> str:
    """Deprecated matrix rewrite; render the upstream-parsed mathematics instead."""
    warnings.warn(
        'str_to_pmatrix is deprecated; matrix notation is handled by Math-Verify', DeprecationWarning, stacklevel=2
    )
    from sympy import latex

    objects, display = _parse(input_str, 'fragment')
    return latex(objects[0]) if len(objects) == 1 else display


def choice_answer_clean(pred: str) -> str:
    """Retain the legacy categorical text cleanup independently of math scoring."""
    pred = pred.strip('\n').rstrip('.').rstrip('/').strip(' ').lstrip(':')
    matches = re.findall(r'\b(A|B|C|D|E)\b', pred.upper())
    return (matches[-1] if matches else pred.strip().strip('.')).rstrip('.').rstrip('/')
