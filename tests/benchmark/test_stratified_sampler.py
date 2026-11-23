"""Stratified collection sampling must honor the requested size and cover available datasets."""

from collections import Counter

import pytest

from evalscope.api.dataset import DatasetDict, MemoryDataset, Sample
from evalscope.collections import CollectionSchema, DatasetInfo, StratifiedSampler


@pytest.mark.parametrize(
    ('sizes', 'budget', 'expected'),
    [
        ([1, 99], 10, [1, 9]),
        ([99, 1], 10, [9, 1]),
        ([5] * 10, 19, [2] * 9 + [1]),
        ([10, 90], 10, [1, 9]),
        ([0, 5, 1, 8], 2, [0, 1, 0, 1]),
        ([2, 0, 3], 10, [2, 0, 3]),
        ([0, 0], 4, [0, 0]),
        ([3, 5], 0, [0, 0]),
    ],
)
def test_stratified_sample_quotas(
    sizes: list[int], budget: int, expected: list[int], monkeypatch: pytest.MonkeyPatch
) -> None:
    sources = {
        str(i): DatasetDict({'default': MemoryDataset([Sample(input=f'{i}/{j}') for j in range(size)])})
        for i, size in enumerate(sizes)
    }
    monkeypatch.setattr(DatasetInfo, 'get_data', lambda dataset: sources[dataset.name])
    schema = CollectionSchema(name='index', datasets=[DatasetInfo(name=name) for name in sources])

    entries = StratifiedSampler(schema).sample(budget)

    counts = Counter(entry['dataset_name'] for entry in entries)
    assert [counts[name] for name in sources] == expected
    assert len(entries) == min(budget, sum(sizes))
    assert len({entry['prompt']['input'] for entry in entries}) == len(entries)
