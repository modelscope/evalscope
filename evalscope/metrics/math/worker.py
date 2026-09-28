"""Module entry point for Math-Verify; math objects exist only in the child."""

import contextlib
import math
import re
import sys
from decimal import Decimal, InvalidOperation, localcontext
from typing import Any

from .contracts import MathRequest, MathResult


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


def _boxed_parts(text: str) -> list[str]:
    from latex2sympy2_extended.math_normalization import extract_boxed_content

    # Preserve complete subquestion boundaries and order before upstream normalization.
    # An unfinished outer answer must not promote an inner box to a new subquestion.
    parts = []
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
                end = stop
                break
        else:
            break
    return parts


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


def handle_request(request: MathRequest) -> MathResult:
    """Extract and grade parsed objects, never fallback strings."""
    if request.operation == 'boxed':
        return MathResult(parts=_boxed_parts(request.prediction))
    predictions, display = _parse(request.prediction, request.prediction_mode)
    result = MathResult(extracted=display, prediction_valid=bool(predictions))
    if request.operation == 'extract':
        return result
    if request.operation == 'number':
        if len(predictions) == 1 and getattr(predictions[0], 'is_number', False) and predictions[0].is_real:
            value = float(predictions[0])
            result.number = value if math.isfinite(value) else None
        return result
    if request.operation == 'render':
        from sympy import latex

        result.extracted = latex(predictions[0]) if len(predictions) == 1 else display
        return result
    golds, gold_display = _parse(request.reference, 'fragment')
    result.reference_valid = bool(golds)
    if not golds:
        result.reason = 'invalid_reference'
        return result
    if request.numeric_reference and (
        len(golds) != 1 or not golds[0].is_number or not golds[0].is_real or not golds[0].is_finite
    ):
        result.reference_valid = False
        result.reason = 'invalid_numeric_reference'
        return result
    if request.integer_only and not _integer_literal(gold_display):
        result.reference_valid = False
        result.reason = 'invalid_integer_reference'
        return result
    if not predictions:
        result.reason = 'prediction_not_parsed'
        return result
    if request.integer_only and not _integer_literal(display):
        result.reason = 'integer_literal_required'
        return result
    if request.multiple_answers:
        golds = _answer_components(golds, request.reference)
        predictions = _answer_components(predictions, request.prediction)
        if not golds:
            result.reference_valid = False
            result.reason = 'invalid_reference'
            return result
        if len(golds) != len(predictions):
            result.reason = 'answer_count_mismatch'
            return result
    absolute = request.absolute_tolerance
    if absolute is not None and len(absolute) not in (1, len(golds)):
        raise ValueError('Tolerance count must be one or equal to the reference answer count')

    def matches(gold: Any, prediction: Any, index: int) -> bool:
        if absolute is not None or request.relative_tolerance is not None:
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
                    request.relative_tolerance or 0,
                )
        return _verify(gold, prediction)

    if request.multiple_answers:

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
    return result


def main() -> None:
    """Serve Pydantic JSON messages without library signal timers."""
    output = sys.stdout
    with contextlib.redirect_stdout(sys.stderr):
        for line in sys.stdin:
            try:
                result = handle_request(MathRequest.model_validate_json(line))
            except Exception as exc:
                result = MathResult(error=f'{type(exc).__name__}: {exc}')
            output.write(result.model_dump_json() + '\n')
            output.flush()


if __name__ == '__main__':
    main()
