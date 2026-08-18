"""Regression tests for dataset and choice shuffling determinism.

The same configuration must rebuild the same dataset: same sample order, same option order and
same answer letter. Otherwise `--use-cache` / `--rerun-review` score cached predictions against a
different answer key.
"""

import hashlib
import logging
import pytest
import random
from typing import List

from evalscope.api.dataset import MemoryDataset, Sample
from evalscope.api.dataset.utils import shuffle_choices_if_requested
from evalscope.utils.io_utils import content_seed

CHOICES = ['alpha', 'beta', 'gamma', 'delta']


@pytest.fixture
def caplog_evalscope(caplog, monkeypatch):
    """evalscope's logger sets ``propagate = False``, so bare ``caplog`` records nothing."""
    monkeypatch.setattr(logging.getLogger('evalscope'), 'propagate', True)
    caplog.set_level(logging.WARNING, logger='evalscope')
    return caplog


def _samples(count: int) -> List[Sample]:
    return [
        Sample(input=f'question {i}', choices=list(CHOICES), target='A', metadata={'i': i}) for i in range(count)
    ]


def _dataset(count: int) -> MemoryDataset:
    dataset = MemoryDataset(samples=_samples(count))
    dataset.reindex()
    return dataset


def _orders(dataset: MemoryDataset) -> List[tuple]:
    return [(tuple(sample.choices), str(sample.target)) for sample in dataset]


def test_content_seed_matches_the_gpqa_legacy_formula() -> None:
    """gpqa delegates to `content_seed`; a drift here would silently re-letter every gpqa answer."""
    text = 'Two quantum states with energies E1 and E2.'

    assert content_seed(text) == int.from_bytes(hashlib.sha256(text.encode('utf-8')).digest()[:8], 'big')


def test_dataset_order_is_reproducible_with_a_seed() -> None:
    first, second = _dataset(20), _dataset(20)

    first.shuffle(seed=42)
    second.shuffle(seed=42)

    assert [s.metadata['i'] for s in first] == [s.metadata['i'] for s in second]


def test_choice_order_is_reproducible() -> None:
    first, second = _dataset(10), _dataset(10)

    first.shuffle_choices(seed=42)
    second.shuffle_choices(seed=42)

    assert _orders(first) == _orders(second)


def test_choice_order_is_reproducible_without_an_explicit_seed() -> None:
    """`shuffle_choices=True` carries no seed of its own, and used to fall back to OS entropy."""
    first, second = _dataset(10), _dataset(10)

    first.shuffle_choices()
    second.shuffle_choices()

    assert _orders(first) == _orders(second)


def test_choice_order_survives_a_global_rng_perturbation() -> None:
    orders = []
    for seed in range(4):
        random.seed(seed)
        dataset = _dataset(6)
        dataset.shuffle_choices(seed=42)
        orders.append(_orders(dataset))

    assert all(order == orders[0] for order in orders)


def test_choice_order_does_not_depend_on_which_samples_were_filtered() -> None:
    """A sample's options must not change because a *different* sample was filtered out.

    `filter_func` runs before the choice shuffle in every loader, so with one RNG advanced over the
    list, dropping one sample re-lettered the answers of every sample after it.
    """
    everything = _dataset(10)
    without_one = MemoryDataset(samples=[s for s in _samples(10) if s.metadata['i'] != 2])

    everything.shuffle_choices(seed=42)
    without_one.shuffle_choices(seed=42)

    survivors = {s.input: (tuple(s.choices), str(s.target)) for s in without_one}
    assert all(
        survivors[s.input] == (tuple(s.choices), str(s.target)) for s in everything if s.metadata['i'] != 2
    )


def test_choice_order_does_not_depend_on_position() -> None:
    forward = MemoryDataset(samples=_samples(6))
    reversed_order = MemoryDataset(samples=list(reversed(_samples(6))))

    forward.shuffle_choices(seed=42)
    reversed_order.shuffle_choices(seed=42)

    by_question = {s.input: (tuple(s.choices), str(s.target)) for s in reversed_order}
    assert all(by_question[s.input] == (tuple(s.choices), str(s.target)) for s in forward)


def test_target_still_points_at_the_original_answer() -> None:
    dataset = _dataset(8)

    dataset.shuffle_choices(seed=42)

    for sample in dataset:
        assert sample.choices[ord(str(sample.target)) - ord('A')] == CHOICES[0]


def test_shuffle_still_varies_between_samples() -> None:
    """Position-bias protection must survive the switch to a content-derived seed."""
    dataset = _dataset(12)

    dataset.shuffle_choices(seed=42)

    assert len({tuple(sample.choices) for sample in dataset}) > 1


def test_run_seed_reaches_the_choice_shuffle() -> None:
    """`shuffle_choices=True` must use the run's seed, not a fixed one."""
    with_42, with_7 = _dataset(12), _dataset(12)

    shuffle_choices_if_requested(with_42, True, 42)
    shuffle_choices_if_requested(with_7, True, 7)

    assert _orders(with_42) != _orders(with_7)


def test_unseeded_dataset_shuffle_warns(caplog_evalscope) -> None:
    dataset = _dataset(10)

    dataset.shuffle()

    assert 'cannot be reproduced' in caplog_evalscope.text
