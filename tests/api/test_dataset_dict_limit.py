from typing import List

from evalscope.api.dataset import MemoryDataset, Sample
from evalscope.api.dataset.dataset import DatasetDict


def _build_dataset(subset_sizes: dict) -> MemoryDataset:
    samples: List[Sample] = []
    for subset_key, size in subset_sizes.items():
        for idx in range(size):
            samples.append(Sample(input=f'{subset_key}-{idx}', subset_key=subset_key))
    return MemoryDataset(samples, name='dummy')


def test_from_dataset_float_limit_applies_per_subset() -> None:
    # Regression test for issue #1525: a float limit must be resolved
    # against each subset's own size, not the first subset's size.
    dataset = _build_dataset({'small': 4, 'large': 10})

    dataset_dict = DatasetDict.from_dataset(dataset, subset_list=['small', 'large'], limit=0.5)

    assert len(dataset_dict['small']) == 2  # 4 * 0.5
    assert len(dataset_dict['large']) == 5  # 10 * 0.5, was 2 before the fix


def test_from_dataset_int_limit_applies_per_subset() -> None:
    dataset = _build_dataset({'small': 4, 'large': 10})

    dataset_dict = DatasetDict.from_dataset(dataset, subset_list=['small', 'large'], limit=3)

    assert len(dataset_dict['small']) == 3
    assert len(dataset_dict['large']) == 3


def test_from_dataset_no_limit_keeps_all_samples() -> None:
    dataset = _build_dataset({'small': 4, 'large': 10})

    dataset_dict = DatasetDict.from_dataset(dataset, subset_list=['small', 'large'], limit=None)

    assert len(dataset_dict['small']) == 4
    assert len(dataset_dict['large']) == 10
