# Copyright (c) Alibaba, Inc. and its affiliates.

import re
from typing import Optional, Tuple

import sympy
from sympy import N
from sympy.parsing.latex import parse_latex

from evalscope.utils.logger import get_logger
from .utils import (
    WarningType,
    find_last_boxed_content as find_boxed_enhanced,
    normalize_string,
    replace_unicode,
    extract_last_integer,
)
from .parse_manual import manual_mapper, complete_mapper

logger = get_logger()


def remove_inner_boxed(text: str) -> str:
    """Remove inner \\boxed{} commands from text."""
    pattern = r"\\(boxed|fbox)\{([^{}]*)\}"
    while re.search(pattern, text):
        text = re.sub(pattern, r"\2", text)
    return text


def find_last_boxed_content(text: str) -> Optional[str]:
    """
    Find the content of the last \\boxed or \\fbox command in text.

    Args:
        text: The text to search

    Returns:
        The content of the last boxed command, or None if not found
    """
    # Simple regex for matching boxed content (doesn't handle nested braces perfectly)
    pattern = r"\\(?:boxed|fbox)\{([^{}]+)\}"
    matches = list(re.finditer(pattern, text))

    if not matches:
        return None

    last_match = matches[-1].group(1)
    return remove_inner_boxed(last_match)


def normalize_latex(text: str) -> str:
    """Normalize LaTeX expressions for comparison."""
    # Remove extra whitespace
    text = re.sub(r"\s+", "", text)

    # Common LaTeX command replacements
    replacements = {
        r"\\frac": "frac",
        r"\\dfrac": "frac",
        r"\\tfrac": "frac",
        r"\\left": "",
        r"\\right": "",
        r"\\,": "",
        r"\\;": "",
        r"\\:": "",
        r"\\!": "",
        r"\\quad": "",
        r"\\qquad": "",
    }

    for old, new in replacements.items():
        text = text.replace(old, new)

    return text


def parse_latex_to_sympy(text: str) -> Optional[any]:
    """
    Parse LaTeX string to SymPy expression.

    Args:
        text: LaTeX string to parse

    Returns:
        SymPy expression or None if parsing fails
    """
    try:
        # Normalize the LaTeX first
        text = normalize_latex(text)

        # Try to parse as LaTeX
        try:
            expr = parse_latex(text)
            return expr
        except:
            pass

        # Try to parse as plain expression
        try:
            expr = sympy.sympify(text)
            return expr
        except:
            pass

        # Try to parse as a number
        try:
            if "/" in text:
                # Try as fraction
                parts = text.split("/")
                if len(parts) == 2:
                    num = sympy.sympify(parts[0])
                    den = sympy.sympify(parts[1])
                    return num / den
            else:
                # Try as number
                return sympy.Number(text)
        except:
            pass

        return None

    except Exception as e:
        logger.debug(f"Failed to parse LaTeX '{text}': {e}")
        return None


def extract_answer(text: str, strict: bool = False, list_answer: bool = False) -> Tuple[Optional[str], WarningType]:
    """
    Enhanced answer extraction with Unicode handling, manual mappings, and nested bracket support.

    Args:
        text: Model's raw output text
        strict: Whether to use strict parsing (must have \\boxed)
        list_answer: Whether to expect list of answers

    Returns:
        Tuple of (extracted_answer, warning_type)
    """
    if not text or len(text) == 0:
        return None, WarningType.MAJOR

    warning = WarningType.NONE

    # Step 1: Unicode replacement
    text, unicode_warning = replace_unicode(text)
    warning = max(warning, unicode_warning)

    # Step 2: Complete mapper for full response (when no \boxed present)
    if text in complete_mapper:
        text = complete_mapper[text]
        warning = WarningType.MAJOR

    # Step 3: Extract boxed content (with recursive pattern matching)
    boxed_content, boxed_warning = find_boxed_enhanced(text, list_answer)
    warning = max(warning, boxed_warning)

    if boxed_content:
        # Check manual mapper for known problematic formats
        if boxed_content in manual_mapper:
            logger.warning(f"Applying manual mapping to: {boxed_content[:50]}...")
            boxed_content = manual_mapper[boxed_content]
            warning = max(warning, WarningType.MAJOR)
        return boxed_content, warning

    # Step 4: Strict mode check
    if strict:
        return None, WarningType.MAJOR

    # Step 5: Fallback - extract last integer
    last_int, fallback_warning = extract_last_integer(text)
    if last_int is not None:
        return str(last_int), WarningType.MAJOR

    return None, WarningType.MAJOR


def parse_answer(answer_str: str) -> Optional[any]:
    """
    Parse and normalize answer string to SymPy expression.

    Args:
        answer_str: Answer string to parse

    Returns:
        Normalized SymPy expression or the original string if parsing fails
    """
    if not answer_str:
        return None

    try:
        # Try to parse as LaTeX first
        parsed = parse_latex_to_sympy(answer_str)
        if parsed is not None:
            return parsed

        # If parsing failed, return the original string
        return answer_str.strip()

    except Exception as e:
        logger.debug(f"Failed to parse answer '{answer_str}': {e}")
        return answer_str.strip()


def check_answers(model_answer: any, gold_answer: any, epsilon: float = 1e-10) -> bool:
    """
    Check if model answer matches gold answer, supporting AnswerList.

    Args:
        model_answer: Parsed model answer
        gold_answer: Parsed gold answer
        epsilon: Tolerance for numerical comparison

    Returns:
        True if answers match, False otherwise
    """
    try:
        # Import AnswerList here to avoid circular dependency
        from .utils import AnswerList

        # If both are None
        if model_answer is None and gold_answer is None:
            return True

        # If one is None
        if model_answer is None or gold_answer is None:
            return False

        # Check if type mismatch: one is list, other is not
        is_model_list = isinstance(model_answer, (list, tuple, AnswerList))
        is_gold_list = isinstance(gold_answer, (list, tuple, AnswerList))
        if is_model_list != is_gold_list:
            return False

        # AnswerList comparison (order-independent)
        if isinstance(model_answer, AnswerList):
            return model_answer.equals(gold_answer)
        elif isinstance(gold_answer, AnswerList):
            return gold_answer.equals(model_answer)

        # String comparison
        if isinstance(model_answer, str) or isinstance(gold_answer, str):
            return str(model_answer).strip() == str(gold_answer).strip()

        # Numerical comparison (absolute + relative error)
        try:
            if not hasattr(model_answer, "equals"):
                err = abs(N(model_answer - gold_answer))
                if err < epsilon:
                    return True
                # Also check relative error
                max_val = max(abs(N(model_answer)), abs(N(gold_answer)), epsilon)
                rel_err = err / max_val
                return rel_err < epsilon
        except:
            pass

        # SymPy equals method
        try:
            if hasattr(model_answer, "equals") and callable(model_answer.equals):
                return bool(model_answer.equals(gold_answer))
        except:
            pass

        # Convert both to SymPy if needed and retry
        if not isinstance(model_answer, (sympy.Basic, sympy.Expr)):
            try:
                model_answer = sympy.sympify(str(model_answer))
            except:
                pass

        if not isinstance(gold_answer, (sympy.Basic, sympy.Expr)):
            try:
                gold_answer = sympy.sympify(str(gold_answer))
            except:
                pass

        # Try symbolic equality
        try:
            if sympy.simplify(model_answer - gold_answer) == 0:
                return True
        except:
            pass

        # Try numerical equality
        try:
            model_val = complex(N(model_answer))
            gold_val = complex(N(gold_answer))
            return abs(model_val - gold_val) < epsilon
        except:
            pass

        # String equality as last resort
        return str(model_answer).strip() == str(gold_answer).strip()

    except Exception as e:
        logger.debug(f"Error checking answers: {e}")
        # Fallback to string comparison
        return str(model_answer).strip() == str(gold_answer).strip()
