# Copyright (c) Alibaba, Inc. and its affiliates.

from typing import ClassVar

from evalscope.api.dataset import DataLoader, MemoryDataset
from evalscope.api.registry import BENCHMARK_REGISTRY
from evalscope.benchmarks.librispeech.librispeech_adapter import LibriSpeechAdapter
from evalscope.config import TaskConfig


class CapturingDataLoader(DataLoader):

    loaded_splits: ClassVar[list[tuple[str, str]]] = []

    def load(self) -> MemoryDataset:
        self.__class__.loaded_splits.append((self.split, self.subset))
        return MemoryDataset(samples=[])


def test_librispeech_loads_each_test_split_as_a_subset() -> None:
    benchmark_meta = BENCHMARK_REGISTRY['librispeech']
    adapter = LibriSpeechAdapter(
        benchmark_meta=benchmark_meta,
        task_config=TaskConfig(datasets=['librispeech']),
    )
    CapturingDataLoader.loaded_splits = []

    adapter.load_subsets(lambda subset: adapter.load_subset(subset, CapturingDataLoader))

    assert benchmark_meta.subset_list == ['test_clean', 'test_other']
    assert benchmark_meta.eval_split is None
    assert benchmark_meta.evaluation_version == 'v1.1'
    assert CapturingDataLoader.loaded_splits == [
        ('test_clean', 'default'),
        ('test_other', 'default'),
    ]
