"""Regression tests for GPQA choice shuffling determinism (issue #1579).

``record_to_sample`` must be a pure function: an unseeded shuffle made the
option order differ on every dataset build, so ``--rerun-review`` scored
cached predictions against a freshly shuffled answer key.
"""

import random
from typing import Any, Dict, List

from evalscope.api.registry import get_benchmark


def _record(question: str) -> Dict[str, Any]:
    return {
        'Question': question,
        'Correct Answer': '10^-4 eV',
        'Incorrect Answer 1': '10^-8 eV',
        'Incorrect Answer 2': '10^-9 eV',
        'Incorrect Answer 3': '10^-11 eV',
    }


QUESTIONS: List[str] = [
    'Two quantum states with energies E1 and E2 have a lifetime of 10^-9 sec and 10^-8 sec.',
    'Which of the following is the energy difference that can be clearly resolved?',
    'A different graduate-level physics question about spectral line widths.',
]


def test_record_to_sample_is_deterministic_across_calls() -> None:
    adapter = get_benchmark('gpqa_diamond')
    record = _record(QUESTIONS[0])

    samples = []
    for seed in range(8):
        # Perturb the global RNG the way a real run does (dataset shuffle, other
        # adapters, model sampling) to prove the choice order does not depend on it.
        random.seed(seed)
        samples.append(adapter.record_to_sample(record))

    targets = {str(sample.target) for sample in samples}
    orders = {tuple(sample.choices) for sample in samples}
    assert len(targets) == 1, f'answer letter drifted across rebuilds: {targets}'
    assert len(orders) == 1, f'choice order drifted across rebuilds: {orders}'


def test_target_still_points_at_the_correct_answer() -> None:
    adapter = get_benchmark('gpqa_diamond')

    for question in QUESTIONS:
        sample = adapter.record_to_sample(_record(question))
        letter_index = ord(str(sample.target)) - ord('A')
        assert sample.choices[letter_index] == '10^-4 eV'


def test_shuffle_still_varies_between_questions() -> None:
    """Position-bias protection must survive the switch to a seeded RNG."""
    adapter = get_benchmark('gpqa_diamond')

    orders = {tuple(adapter.record_to_sample(_record(question)).choices) for question in QUESTIONS}
    assert len(orders) > 1, 'every question produced the same option order'
