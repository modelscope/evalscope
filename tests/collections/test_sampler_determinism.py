"""Regression tests for collection sampling determinism.

An unseeded `random.sample` made the composition of a mixed dataset differ on every run, so a
collection could not be rebuilt from its schema.
"""

from dataclasses import dataclass, field
from typing import Dict, List

from evalscope.api.dataset import Sample
from evalscope.collections.sampler import UniformSampler, WeightedSampler


@dataclass
class _FakeDatasetInfo:
    """`DatasetInfo` without the benchmark registry: only `get_data` differs."""

    name: str
    weight: float = 1.0
    task_type: str = 'test'
    tags: List[str] = field(default_factory=list)
    hierarchy: List[str] = field(default_factory=list)

    def get_data(self) -> Dict[str, List[Sample]]:
        return {'default': [Sample(input=f'{self.name} question {i}', target='A') for i in range(20)]}


def _drawn(sampler_cls, seed=None) -> List[str]:
    dataset = _FakeDatasetInfo(name='alpha')
    sampler = sampler_cls(schema=None) if seed is None else sampler_cls(schema=None, seed=seed)
    return [entry.prompt['input'] for entry in sampler._sample_dataset(dataset, 8)]


def test_sampling_is_reproducible_by_default() -> None:
    assert _drawn(WeightedSampler) == _drawn(WeightedSampler)


def test_seed_changes_the_drawn_subset() -> None:
    assert _drawn(WeightedSampler, seed=1) != _drawn(WeightedSampler, seed=2)


def test_the_same_seed_agrees_across_sampler_types() -> None:
    """The draw depends on the seed and the dataset, not on which sampler allocated the count."""
    assert _drawn(WeightedSampler, seed=7) == _drawn(UniformSampler, seed=7)


def test_draw_does_not_depend_on_position_in_the_schema() -> None:
    """Each dataset is seeded from its own name, so reordering the schema cannot change its draw."""
    sampler = WeightedSampler(schema=None)
    first, second = _FakeDatasetInfo(name='alpha'), _FakeDatasetInfo(name='beta')

    forward = [[e.prompt['input'] for e in sampler._sample_dataset(d, 8)] for d in (first, second)]
    backward = [[e.prompt['input'] for e in sampler._sample_dataset(d, 8)] for d in (second, first)]

    assert forward == list(reversed(backward))
