"""LaTeX-spaced thousands groups and currency prefixes are read as the number they write (issue #2034)."""

import pytest

from evalscope.api.dataset import Sample
from evalscope.api.evaluator import TaskState
from evalscope.api.model import ModelOutput
from evalscope.api.registry import get_benchmark
from evalscope.config import TaskConfig
from evalscope.metrics.math.parser import _normalize_number_formatting, extract_answer, math_equal

MATCHES = [
    (r'\boxed{70\,000}', '70000'),
    (r'\boxed{70\ 000}', '70000'),
    (r'\boxed{70\;000}', '70000'),
    (r'\boxed{₹70,000}', '70000'),
    (r'\boxed{₹70000}', '70000'),
    (r'\boxed{\text{₹}70,000}', '70000'),
    (r'\boxed{Rs. 70000}', '70000'),
    (r'\boxed{INR 70000}', '70000'),
    (r'\boxed{€70,000}', '70000'),
    (r'\boxed{£70,000}', '70000'),
    (r'\boxed{¥70000}', '70000'),
    (r'\boxed{1\,000\,000}', '1000000'),
    (r'\boxed{1\,250}', '1250'),
    # Controls that already matched before normalization.
    (r'\boxed{70000}', '70000'),
    (r'\boxed{70,000}', '70000'),
    (r'\boxed{70{,}000}', '70000'),
    (r'\boxed{\$70,000}', '70000'),
    (r'\boxed{70000 rupees}', '70000'),
]

# Math-Verify used to combine the groups arithmetically, so these wrong answers matched.
MISMATCHES = [
    (r'\boxed{1\,250}', '251'),
    (r'\boxed{5\,000}', '0'),
    (r'\boxed{70\,000}', '0'),
]


def gsm8k_score(answer: str, gold: str) -> float:
    adapter = get_benchmark('gsm8k', TaskConfig(datasets=['gsm8k'], judge={'strategy': 'rule'}))
    completion = f'So the answer is {answer}'
    state = TaskState(
        model='offline',
        sample=Sample(input='question', target=gold),
        output=ModelOutput.from_content('offline', completion),
        completed=True,
    )
    extracted = adapter.extract_answer(completion, state)
    return adapter.match_score(completion, extracted, gold, state).main_value


@pytest.mark.parametrize(('answer', 'gold'), MATCHES)
def test_formatted_number_matches_gold(answer: str, gold: str) -> None:
    assert math_equal(extract_answer(f'So the answer is {answer}'), gold)
    assert gsm8k_score(answer, gold) == 1.0


@pytest.mark.parametrize(('answer', 'gold'), MISMATCHES)
def test_spaced_groups_no_longer_match_arithmetic_result(answer: str, gold: str) -> None:
    assert not math_equal(extract_answer(f'So the answer is {answer}'), gold)
    assert gsm8k_score(answer, gold) == 0.0


@pytest.mark.parametrize('text', [r'\boxed{2\,3}', r'\boxed{12\,3456}', r'x\,y', r'\boxed{\$70,000}', 'Rsx 5'])
def test_normalization_leaves_other_text_unchanged(text: str) -> None:
    # Only a complete three-digit group is joined, and only listed currency prefixes are dropped.
    assert _normalize_number_formatting(text) == text
