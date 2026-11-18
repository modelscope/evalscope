# flake8: noqa: E501
"""
Utility functions for TIR-Bench evaluation.
Adapted from https://github.com/agents-x-project/TIR-Bench/blob/main/tools.py
"""

import re
from typing import List, Optional, Tuple

from evalscope.utils.logger import get_logger
from evalscope.utils.multi_choices import extract_answer_line

logger = get_logger()

# Valid MCQ choice letters
_CHOICES = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']

_OPTION_LINE_RE = re.compile(r'^[ \t]*([A-J])(?:[.:)]|[ \t]+)[ \t]*(\S[^\n]*)$', re.MULTILINE)
_OPTION_INLINE_RE = re.compile(r'\(([A-J])\)[ \t]*(.*?)(?=\([A-J]\)|$)', re.MULTILINE)

_SINGLE_CHOICE_RE = re.compile(r'(?:[A-J]|\([A-J]\)|\[[A-J]\]|\\boxed\{[A-J]\}|\$[A-J]\$)[.。]?')

# ---------------------------------------------------------------------------
# Answer-type classification
# ---------------------------------------------------------------------------


def classify_string(s: str) -> int:
    """Classify answer string into one of four types.

    Returns:
        1 - alphabetic (MCQ letter answer)
        2 - integer
        3 - float
        4 - unknown / unsupported
    """
    s = str(s).strip()
    if s.isalpha():
        return 1
    try:
        int(s)
        return 2
    except ValueError:
        pass
    try:
        float(s)
        return 3
    except ValueError:
        pass
    return 4


# ---------------------------------------------------------------------------
# MCQ answer extraction (evalscope-specific)
# ---------------------------------------------------------------------------


def is_single_choice_answer(reference: str) -> bool:
    """Return whether the reference is a single choice label."""
    return reference in _CHOICES


def _extract_single_choice(prediction: str) -> str:
    """Read a structured answer or retain the text for exact option matching."""
    answer = prediction.strip()
    last_line = answer.rsplit('\n', 1)[-1]
    if last_line.partition(':')[0].strip(' \t*').casefold() == 'answer':
        answer = extract_answer_line(last_line) or answer
    if _SINGLE_CHOICE_RE.fullmatch(answer):
        return answer.removeprefix(r'\boxed{').strip('()[]{}$.。')
    return answer


def extract_mcq_answer(prediction: str) -> str:
    """Extract a choice using the legacy rules for non-single-choice references."""
    if not prediction:
        return prediction

    # Pattern 1: LaTeX \boxed{D}  — highest priority
    m = re.search(r'\\boxed\{([A-J])\}', prediction)
    if m:
        return m.group(1).upper()

    # Pattern 2: explicit keyword markers
    explicit_patterns = [
        r'(?:ANSWER|Answer|answer)\s*[:\-]\s*([A-J])\b',
        r'(?:The answer is|the answer is|答案[是为])\s*([A-J])\b',
        r'\b([A-J])\s*(?:is correct|is the answer)',
    ]
    for pat in explicit_patterns:
        m = re.search(pat, prediction)
        if m:
            return m.group(1).upper()

    # Pattern 3: parenthesised letter (A) — take the LAST match so that option
    # listings like "(A) cat  (B) dog  …" do not shadow the model's final answer
    paren_matches = re.findall(r'\(([A-J])\)', prediction)
    if paren_matches:
        return paren_matches[-1].upper()

    # Pattern 4: last isolated letter A-J (word boundary)
    matches = re.findall(r'\b([A-J])\b', prediction)
    if matches:
        return matches[-1].upper()

    return prediction.strip()


def extract_answer_with_classify(prediction: str, reference: str) -> str:
    """Dispatch answer extraction based on the type of the reference answer.

    Uses :func:`classify_string` on the reference to select the right strategy:

    - **type 1** (alphabetic / MCQ letter): use explicit answer extraction for
      single-choice items and the legacy parser for other items.
    - **type 2 / 3 / 4** (integer / float / composite like ``[2, 3]``): return
      the raw prediction unchanged so that the task-specific scoring functions
      in :meth:`TIRBenchAdapter.match_score` (``judge_int``, ``judge_float``,
      ``extract_two_numbers``, ``extract_consecutive_integers``, etc.) can do
      their own targeted extraction.

    Args:
        prediction: Raw model output string.
        reference: Ground-truth answer string.

    Returns:
        Extracted or raw prediction string.
    """
    if is_single_choice_answer(reference):
        return _extract_single_choice(prediction)
    if classify_string(reference) == 1:
        return extract_mcq_answer(prediction)
    return prediction


# ---------------------------------------------------------------------------
# MCQ scoring
# ---------------------------------------------------------------------------


def judge_choice(extracted_answer: str, answer: str, prompt_text: str) -> float:
    """Score a single label or exact option text, or compare a multi-choice answer."""
    if not is_single_choice_answer(answer):
        return float(sorted(answer) == sorted(extracted_answer))

    extraction = extracted_answer.removeprefix('Extracted answer:').strip()
    if extraction in _CHOICES:
        return float(extraction == answer)
    if not extraction:
        return 0.0

    matching_labels = {
        label
        for pattern in (_OPTION_LINE_RE, _OPTION_INLINE_RE)
        for label, option in pattern.findall(prompt_text)
        if option.strip() == extraction
    }
    return float(matching_labels == {answer})


# ---------------------------------------------------------------------------
# Integer / float scoring
# ---------------------------------------------------------------------------


def judge_int(extracted_answer: str, answer: str) -> float:
    """Compare the extracted numeric answer with Math-Verify."""
    from evalscope.metrics.math.parser import compare_answers

    prediction = extracted_answer.replace('Extracted answer:', '').strip()
    return float(compare_answers(prediction, str(answer), prediction_mode='output').matched)


def judge_float(extracted_answer: str, answer: str) -> float:
    """Keep the public float-scoring entry point on the same numeric judge."""
    return judge_int(extracted_answer, answer)


# ---------------------------------------------------------------------------
# Jigsaw task helpers
# ---------------------------------------------------------------------------


def extract_consecutive_n_squared(s: str, n: int) -> List[int]:
    """Extract the first sequence of exactly n² consecutive integers from string s.

    Args:
        s: Input string (mixed content).
        n: Grid dimension; target is n² numbers.

    Returns:
        List of n² integers.

    Raises:
        ValueError: If no such subsequence is found.
    """
    total = n * n
    sequences = re.findall(r'[\d\s,，]+', s)
    for seq in sequences:
        nums = re.findall(r'\d+', seq)
        if len(nums) == total:
            return [int(num) for num in nums]
    raise ValueError(f'No consecutive sequence of {total} numbers found in: {s!r}')


def compare(l1: List[int], l2: List[int]) -> float:
    """Compute position-wise accuracy between two lists."""
    if not l1 or len(l1) != len(l2):
        return 0.0
    correct = sum(1 for a, b in zip(l1, l2) if a == b)
    return correct / len(l1)


# ---------------------------------------------------------------------------
# Spot-difference task helpers
# ---------------------------------------------------------------------------


def extract_consecutive_integers(s: str) -> List[int]:
    """Extract all integers from a string (allowing comma/space separators)."""
    matches = re.findall(r'\d+(?=[,\s]*|\b)', s)
    return [int(num) for num in matches]


def list_iou(l_response: List[int], l_answer: List[int]) -> float:
    """Compute Intersection-over-Union between two integer lists (treated as sets)."""
    set_response = set(l_response)
    set_answer = set(l_answer)
    intersection = set_response & set_answer
    union = set_response | set_answer
    if not union:
        return 1.0  # Both empty → perfect match
    return len(intersection) / len(union)


# ---------------------------------------------------------------------------
# Word-search task helpers
# ---------------------------------------------------------------------------


def extract_two_numbers(text: str) -> Optional[Tuple[int, int]]:
    """Extract two consecutive comma-separated integers from text.

    Returns:
        Tuple (num1, num2) or None if not found.
    """
    match = re.search(r'\b(\d+)\s*,\s*(\d+)\b', text)
    if match:
        return int(match.group(1)), int(match.group(2))
    return None
