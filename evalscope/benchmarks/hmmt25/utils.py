"""
HMMT25 utility functions ported from reference implementation.
Provides enhanced answer extraction, parsing, and comparison capabilities.
"""

import re
from enum import Enum
from functools import total_ordering
from typing import Any, Optional, Tuple

try:
    import regex
    USE_REGEX = True
except ImportError:
    import re as regex
    USE_REGEX = False

from evalscope.utils.logger import get_logger
from .parse_manual import manual_mapper, complete_mapper

logger = get_logger()


@total_ordering
class WarningType(Enum):
    """Four-level warning system for diagnostics"""
    NONE = 0
    MINOR = 1
    POSSIBLE = 2
    MAJOR = 3

    def __lt__(self, other):
        if self.__class__ is other.__class__:
            return self.value < other.value
        return self.value < other


def strip(s: str) -> str:
    """Custom strip function that handles LaTeX-specific whitespace"""
    s = s.strip()
    # Remove \\n sequences
    while s.startswith(r"\n"):
        s = s[2:]
    while s.endswith(r"\n"):
        s = s[:-2]
    # Remove \\ sequences
    while s.startswith("\\ "):
        s = s[2:]
    # Remove leading escape sequences before brackets
    while re.match(r"\\{2,}\n?\(", s):
        s = s[3:]
    return s


def remove_aligns(s: str) -> str:
    """Remove LaTeX align environments from string"""
    pattern = r"\\begin{align[^}]*}(.*?)\\end{align[^}]*}"
    return re.sub(pattern, lambda m: m.group(1).replace("&", "").replace("\\\\", ""), s, flags=re.DOTALL)


def remove_invalid_characters(text: str) -> str:
    """Remove invalid LaTeX spacing characters"""
    text = re.sub(r"\\;", "", text)
    text = re.sub(r"\\:", "", text)
    text = re.sub(r"\\,", "", text)
    text = re.sub(r"\\!", "", text)
    return text


def remove_outer_brackets(s: str) -> str:
    """Remove outermost matching brackets if they encompass entire string"""
    while True:
        if not s:
            return s
        opening = s[0]
        closing = s[-1]

        if opening == "(" and closing == ")":
            count = 0
            matched = True
            for i, char in enumerate(s):
                if char == opening:
                    count += 1
                elif char == closing:
                    count -= 1
                if count == 0 and i != len(s) - 1:
                    matched = False
                    break

            if matched:
                s = s[1:-1]
                continue
        break

    return s


def replace_unicode(text: str) -> Tuple[str, WarningType]:
    """Replace unicode characters with LaTeX equivalents"""
    text_old = text

    # Critical replacements (affect boxed detection)
    text = text.replace("\u23a7", r"\boxed{")
    text = text.replace("\u23ab", r"}")
    text = text.replace("\n\u2502", r"\boxed{")
    text = text.replace("\u2502", r"}")
    text = text.replace("\n\u2503", r"\boxed{")
    text = text.replace("\u2503", r"}")
    text = text.replace("\n\uf8f0", r"\boxed{")
    text = text.replace("\uf8fb", r"}")

    warning = WarningType.NONE if text == text_old else WarningType.POSSIBLE

    # Standard replacements (always safe)
    text = text.replace("\u221a", r"\sqrt")
    text = text.replace("\u00d7", r"\cdot")
    text = text.replace("\u202f", r" ")
    text = text.replace("\u2212", "-")
    text = text.replace("\u03c0", r"\pi")

    return text, warning


def remove_inner_boxed(match: str) -> str:
    """Remove inner \\boxed or \\fbox commands from string"""
    if not USE_REGEX:
        # Simple fallback - just return as is
        return match

    pattern = r"(\\boxed|\\fbox)\{((?:[^{}]|\{(?2)\})*)\}"
    matches = list(regex.finditer(pattern, match))
    if not matches:
        return match
    for m in matches:
        match = match.replace(m.group(0), m.group(2))
    return match


def find_last_boxed_content(text: str, list_answer: bool = False) -> Tuple[Optional[str], WarningType]:
    """
    Find content of last \\boxed or \\fbox command using recursive pattern matching.

    Args:
        text: Text to search
        list_answer: Whether to expect list of answers

    Returns:
        Tuple of (extracted content, warning level)
    """
    if USE_REGEX:
        # Use recursive pattern matching: (?2) references capture group 2
        pattern = r"(boxed|fbox)\{((?:[^{}]|\{(?2)\})*)\}"
    else:
        # Fallback to simple pattern (won't handle nested brackets correctly)
        pattern = r"(boxed|fbox)\{([^{}]+)\}"
        logger.warning("regex library not available, nested brackets may fail")

    matches = list(regex.finditer(pattern, text))
    if not matches:
        return None, WarningType.NONE

    # List answer special handling: multiple boxed on same line
    if len(matches) > 1 and list_answer:
        split_text = text.split("\n")
        for i in range(len(split_text) - 1, -1, -1):
            matches_line = list(regex.finditer(pattern, split_text[i]))
            if len(matches_line) > 0:
                returned_boxed = ",".join([match.group(2) for match in matches_line])
                return remove_inner_boxed(returned_boxed), WarningType.POSSIBLE

    last_match = remove_inner_boxed(matches[-1].group(2))
    return last_match, WarningType.NONE


def replace_and_or(s: str) -> str:
    """
    Replace 'and' or 'or' with commas in list answers.

    Rules:
    1) If 'and/or' (or \\text{} forms) is NOT next to a comma -> replace with ','
    2) Otherwise (comma already present) -> delete it
    """
    TOKEN = re.compile(
        r"""
        (?:\\text\s*\{\s*)?      # optional '\\text{' and any leading blanks
        (and|or)                 # the word itself
        (?:\s*\})?               # optional closing '}' with any blanks
        """,
        re.I | re.VERBOSE,
    )

    out, idx = [], 0
    for m in TOKEN.finditer(s):
        start, end = m.span()
        # copy text before the token
        out.append(s[idx:start])

        # look to left of token, skipping blanks
        j = start - 1
        while j >= 0 and s[j].isspace():
            j -= 1
        comma_left = j >= 0 and s[j] == ","

        # look to right of token, skipping blanks
        k = end
        while k < len(s) and s[k].isspace():
            k += 1
        comma_right = k < len(s) and s[k] == ","

        # choose replacement
        out.append("" if (comma_left or comma_right) else ",")
        idx = end

    out.append(s[idx:])
    return "".join(out)


def normalize_string(s: str, list_answer: bool = False) -> str:
    """
    Normalize LaTeX string for parsing using 50+ standardization rules.

    Args:
        s: LaTeX string to normalize
        list_answer: Whether this is a list answer

    Returns:
        Normalized string
    """
    # Remove LaTeX size commands
    s = s.replace(r"\left", "").replace(r"\right", "")
    s = s.replace(r"\Bigl", "").replace(r"\Bigr", "")
    s = s.replace(r"\bigl", "").replace(r"\bigr", "")
    s = s.replace(r"\Big", "").replace(r"\big", "")
    s = s.replace(r"\Large", "").replace(r"\large", "")

    # Remove align environments
    s = remove_aligns(s)

    # Bracket standardization
    s = s.replace("[", "(").replace("]", ")")
    s = s.replace("\\{", "(")  # sets → lists
    s = s.replace("\\}", ")")

    # Remove dollar signs and spacing
    s = s.replace("$", "")
    s = s.replace("\\ ", " ")
    s = s.replace(r"\hline", "")
    s = s.replace(r"\vline", "")
    s = s.replace(r"\quad", " ")

    # Unicode-like characters
    s = s.replace("−", "-")
    s = s.replace("–", "-")
    s = s.replace("·", " \\cdot ")

    # Angle notation
    s = s.replace("^\\circ", " ")
    s = s.replace("^{\\circ}", " ")

    # Display style
    s = s.replace("\\displaystyle", "")
    s = s.replace("\\(", "(")
    s = s.replace("\\)", ")")

    # Remove artifacts
    s = s.replace("{,}", "")

    # Remove trailing period
    if s.endswith("."):
        s = s[:-1]

    # List answer handling
    if list_answer and s is not None:
        s = replace_and_or(s)

    # Comma handling
    if not list_answer:
        # Remove thousand separators: 1,000 → 1000
        s = re.sub(r"(?<=\d),(?=\d)", "", s)
        s = s.replace("{,}", "")
    else:
        s = s.replace(";", ",")
        s = s.replace("{,}", ",")

    # Fix \\sqrt spacing
    if "\\sqrt " in s:
        s = re.sub(r"\\sqrt\s*([^\s{}]*)", r"\\sqrt{\1}", s)

    # Remove text content
    s = re.sub(r"\\text\{.*?\}", "", s)

    # Replace mathrm
    s = re.sub(r"\\mathrm\{(.*?)\}", r" \1 ", s)

    # Special constant (Fibonacci F_30)
    s = s.replace("F_{30}", "832040")

    # Extract from equations
    if "=" in s:
        s = s.split("=")[-1]
    if r"\in" in s and list_answer:
        s = s.split(r"\in")[-1]

    # Handle approximations
    if "\\approx" in s:
        s = s.split("\\approx")[0]
        if s.endswith("("):
            s = s[:-1]

    return strip(s)


def extract_last_integer(text: str) -> Tuple[Optional[int], WarningType]:
    """
    Extract last integer from text as fallback.

    Args:
        text: Text to search

    Returns:
        Tuple of (last integer, warning level)
    """
    if not USE_REGEX:
        pattern = r"\b\d+\b"
        matches = list(re.finditer(pattern, text))
    else:
        pattern = r"\b\d+\b"
        matches = list(regex.finditer(pattern, text))

    if not matches:
        return None, WarningType.MAJOR

    try:
        return int(matches[-1].group()), WarningType.MAJOR
    except Exception as e:
        logger.warning(f"Error extracting last integer: {e}")
        return None, WarningType.MAJOR


class AnswerList:
    """
    Answer list class supporting order-independent comparison.
    Uses bipartite matching algorithm for equality checking.
    """

    def __init__(self, answers):
        """
        Initialize AnswerList, filtering out invalid answers.

        Args:
            answers: List or tuple of answers
        """
        if not isinstance(answers, (list, tuple)):
            raise ValueError(f"Expected list/tuple, got {type(answers)}")

        # Only keep answers that contain at least one digit
        valid_answers = []
        for answer in answers:
            if bool(re.search(r"\d", str(answer))):
                valid_answers.append(answer)
            else:
                logger.warning(f"Filtered out answer without numbers: {answer}")

        self.answers = list(valid_answers)

    def equals(self, other) -> bool:
        """
        Check equality with another list (order-independent).
        Uses greedy bipartite matching algorithm.

        Args:
            other: Another AnswerList, list, or tuple

        Returns:
            True if lists match (regardless of order)
        """
        if not isinstance(other, (list, tuple, AnswerList)):
            return False

        other_list = other.answers if isinstance(other, AnswerList) else other

        if len(self.answers) != len(other_list):
            return False

        # Import check_answers here to avoid circular dependency
        from .parser import check_answers

        # Greedy matching: find unique match for each answer
        matched_indices = set()
        for ans1 in self.answers:
            match_found = False
            for i, ans2 in enumerate(other_list):
                if i not in matched_indices and check_answers(ans1, ans2):
                    matched_indices.add(i)
                    match_found = True
                    break
            if not match_found:
                return False

        return True

    def __str__(self):
        return "[" + ",".join([str(ans) for ans in self.answers]) + "]"

    def __len__(self):
        return len(self.answers)

    def __iter__(self):
        return iter(self.answers)


def parse_single_answer(s: str, primitive_type=None) -> Tuple[Any, WarningType]:
    """
    Parse a single answer string.

    Args:
        s: String to parse
        primitive_type: Optional primitive type hint

    Returns:
        Tuple of (parsed answer, warning level)
    """
    import sympy
    from sympy.parsing.latex import parse_latex

    # Try integer
    if s.isdigit():
        return int(s), WarningType.NONE

    # Try float
    try:
        val = float(s)
        if int(val) == val:
            return int(val), WarningType.NONE
        return val, WarningType.NONE
    except:
        pass

    # Try LaTeX parsing
    try:
        # Normalize first
        normalized = normalize_string(s, list_answer=False)

        # Try parse_latex
        try:
            expr = parse_latex(normalized)
            return expr, WarningType.NONE
        except:
            pass

        # Try sympify
        try:
            expr = sympy.sympify(normalized)
            return expr, WarningType.NONE
        except:
            pass

        # Return as string if all parsing fails
        return s.strip(), WarningType.MAJOR

    except Exception as e:
        logger.debug(f"Failed to parse '{s}': {e}")
        return s.strip(), WarningType.MAJOR


def parse_answer(s: str, primitive_type=None, list_answer: bool = False) -> Tuple[Any, WarningType]:
    """
    Parse answer string to SymPy expression, supporting list answers.

    Args:
        s: Answer string to parse
        primitive_type: Optional primitive type hint
        list_answer: Whether to expect list of answers

    Returns:
        Tuple of (parsed answer, warning level)
    """
    warning = WarningType.NONE

    # Apply manual mapper
    if s in manual_mapper:
        logger.warning(f"Applying manual mapping: {s[:50]}...")
        s = manual_mapper[s]
        warning = WarningType.MAJOR

    # Normalize string
    s = normalize_string(s, list_answer)

    # Check if this looks like a list (contains comma)
    if list_answer or ',' in s:
        parts = [p.strip() for p in s.split(',')]
        parsed_parts = []
        max_warning = warning

        for part in parts:
            if not part:  # Skip empty parts
                continue

            parsed, part_warning = parse_single_answer(part, primitive_type)
            if parsed is not None:
                parsed_parts.append(parsed)
                max_warning = max(max_warning, part_warning)

        if len(parsed_parts) > 1:
            # Multiple answers - return as AnswerList
            return AnswerList(parsed_parts), max_warning
        elif len(parsed_parts) == 1:
            # Single answer
            return parsed_parts[0], max_warning
        else:
            # No valid parsed parts
            return None, WarningType.MAJOR

    # Single answer (no comma)
    return parse_single_answer(s, primitive_type)
