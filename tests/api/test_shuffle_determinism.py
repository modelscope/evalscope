"""Regression tests for deterministic choice shuffling."""

import hashlib

from evalscope.api.dataset import MemoryDataset, Sample
from evalscope.api.dataset.utils import shuffle_choices_if_requested
from evalscope.utils.io_utils import content_seed

CHOICES = ['alpha', 'beta', 'gamma', 'delta']


def _dataset(count: int) -> MemoryDataset:
    return MemoryDataset(
        samples=[Sample(input=f'question {index}', choices=list(CHOICES), target='A') for index in range(count)]
    )


def _answers_by_question(dataset: MemoryDataset) -> dict[str, tuple[tuple[str, ...], str]]:
    return {
        str(sample.input): (tuple(sample.choices or []), str(sample.target))
        for sample in dataset
    }


def test_content_seed_is_stable() -> None:
    expected = int.from_bytes(hashlib.sha256(b'one\x00two').digest()[:8], 'big')

    assert content_seed('one', 'two') == expected


def test_choice_shuffle_uses_the_run_seed() -> None:
    with_42 = _dataset(8)
    with_7 = _dataset(8)

    shuffle_choices_if_requested(with_42, True, seed=42)
    shuffle_choices_if_requested(with_7, True, seed=7)

    assert _answers_by_question(with_42) != _answers_by_question(with_7)


def test_choice_shuffle_is_reproducible_and_filter_independent() -> None:
    all_samples = _dataset(6)
    filtered_samples = MemoryDataset(samples=[sample for sample in _dataset(6) if sample.input != 'question 1'])
    repeat = _dataset(6)

    all_samples.shuffle_choices(seed=42)
    filtered_samples.shuffle_choices(seed=42)
    repeat.shuffle_choices(seed=42)

    all_answers = _answers_by_question(all_samples)
    filtered_answers = _answers_by_question(filtered_samples)
    assert all_answers == _answers_by_question(repeat)
    assert filtered_answers == {
        question: answer for question, answer in all_answers.items() if question != 'question 1'
    }


def test_choice_shuffle_remaps_the_correct_answer() -> None:
    dataset = _dataset(8)

    dataset.shuffle_choices(seed=42)

    for sample in dataset:
        answer_index = ord(str(sample.target)) - ord('A')
        assert sample.choices[answer_index] == 'alpha'
