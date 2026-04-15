# flake8: noqa: E501
"""
Utility functions for TIR-Bench evaluation.
Adapted from https://github.com/agents-x-project/TIR-Bench/blob/main/tools.py
"""
import re
from typing import List, Optional, Tuple

from evalscope.utils.logger import get_logger

logger = get_logger()

# Valid MCQ choice letters
_CHOICES = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']

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


def extract_mcq_answer(prediction: str) -> str:
    """Extract a MCQ choice letter (A-J) from a model prediction.

    Priority order:
    1. \\boxed{X} – LaTeX boxed answer (highest priority)
    2. Explicit keyword patterns: 'ANSWER: A', 'Answer: A', 'The answer is A', etc.
    3. '(X)' parenthesised letter – take the **last** occurrence to avoid matching
       option listings in the reasoning text
    4. Last standalone uppercase letter A-J at a word boundary
    5. Return the stripped prediction as-is if nothing matches
    """
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

    - **type 1** (alphabetic / MCQ letter): apply :func:`extract_mcq_answer`
      to pull a single choice letter out of the model's reasoning text.
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
    if classify_string(reference) == 1:
        return extract_mcq_answer(prediction)
    return prediction


# ---------------------------------------------------------------------------
# Levenshtein-based nearest match
# ---------------------------------------------------------------------------


def get_most_similar(prediction: str, choices: List[str]) -> str:
    """Return the choice most similar to prediction using Levenshtein distance."""
    from evalscope.metrics.metrics import levenshtein_distance
    distances = [levenshtein_distance(prediction, choice) for choice in choices]
    return choices[distances.index(min(distances))]


# ---------------------------------------------------------------------------
# MCQ scoring
# ---------------------------------------------------------------------------


def judge_choice(extracted_answer: str, answer: str, prompt_text: str) -> float:
    """Score an MCQ response against the ground-truth answer.

    Handles single-choice (len(answer)==1) and multi-choice questions.
    Uses Levenshtein nearest-match as a fallback when extracted_answer is
    not a valid choice letter.
    """
    correctness = 0.0

    if len(answer) == 1 and answer in _CHOICES:
        # Single-choice question
        extraction = extracted_answer.replace('Extracted answer:', '').strip()
        if extraction not in _CHOICES:
            # Try to match via prompt context (e.g. "A. cat" style options)
            for x in _CHOICES:
                if (
                    f'{x}. {extraction}' in prompt_text or f'{x}.{extraction}' in prompt_text
                    or f'{x} {extraction}' in prompt_text or f'{x}{extraction}' in prompt_text
                ):
                    extraction = x
                    break
            extraction = get_most_similar(extraction, _CHOICES)
        if extraction == answer:
            correctness = 1.0
    else:
        # Multi-choice question: compare sorted letter sets
        sorted_answer = ''.join(sorted(answer))
        sorted_extracted = ''.join(sorted(extracted_answer))
        if sorted_answer == sorted_extracted:
            correctness = 1.0

    return correctness


# ---------------------------------------------------------------------------
# Integer / float scoring
# ---------------------------------------------------------------------------


def judge_int(extracted_answer: str, answer: str) -> float:
    """Score an integer answer."""
    correctness = 0.0
    extraction = extracted_answer.replace('Extracted answer:', '').strip()
    extraction = re.sub(r'[A-Za-z*:\s]+', '', extraction).strip()
    try:
        if int(extraction) == int(answer):
            correctness = 1.0
    except (ValueError, TypeError):
        pass

    if correctness == 0.0:
        # Fallback: use math_equal for symbolic/numeric equivalence
        try:
            from evalscope.metrics.math_parser import math_equal
            if math_equal(extracted_answer, str(answer)):
                correctness = 1.0
        except Exception:
            pass

    return correctness


def judge_float(extracted_answer: str, answer: str) -> float:
    """Score a floating-point answer."""
    correctness = 0.0
    extraction = extracted_answer.replace('Extracted answer:', '').strip()
    extraction = re.sub(r'[A-Za-z*:\s]+', '', extraction).strip()
    try:
        if float(extraction) == float(answer):
            correctness = 1.0
    except (ValueError, TypeError):
        pass

    if correctness == 0.0:
        try:
            from evalscope.metrics.math_parser import math_equal
            if math_equal(extracted_answer, str(answer)):
                correctness = 1.0
        except Exception:
            pass

    return correctness


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
