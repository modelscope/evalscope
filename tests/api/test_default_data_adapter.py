import copy
from typing import Any, ClassVar, List, Optional

from evalscope.api.benchmark import BenchmarkMeta, DefaultDataAdapter
from evalscope.api.dataset import DataLoader, DatasetDict, MemoryDataset, Sample
from evalscope.config import TaskConfig


class CapturingDataLoader(DataLoader):

    latest_limit: ClassVar[Optional[int]] = None
    latest_repeats: ClassVar[Optional[int]] = None

    def load(self) -> MemoryDataset:
        self.__class__.latest_limit = self.limit
        self.__class__.latest_repeats = self.repeats

        samples: List[Sample] = [
            Sample(input='question-1', target='answer-1', subset_key='subset-a'),
            Sample(input='question-2', target='answer-2', subset_key='subset-a'),
        ]
        if self.limit is not None:
            samples = samples[:self.limit]
        if self.repeats > 1:
            samples = [copy.deepcopy(sample) for sample in samples for _ in range(self.repeats)]

        dataset = MemoryDataset(samples=samples, name='dummy')
        dataset.reindex(group_size=self.repeats)
        return dataset


class DummyReformatAdapter(DefaultDataAdapter):

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.reformat_subset = True

    def record_to_sample(self, record: Any) -> Sample:
        return Sample(input=str(record), target='', subset_key='subset-a')


def make_adapter(repeats: int = 3, limit: Optional[int] = None) -> DummyReformatAdapter:
    task_config = TaskConfig(datasets=['dummy'], repeats=repeats, limit=limit)
    benchmark_meta = BenchmarkMeta(
        name='dummy',
        dataset_id='dummy',
        subset_list=['subset-a'],
        default_subset='default',
        eval_split='test',
    )
    return DummyReformatAdapter(benchmark_meta=benchmark_meta, task_config=task_config)


def test_reformat_subset_repeats_are_applied_once_after_grouping() -> None:
    adapter = make_adapter(repeats=3)

    dataset_dict: DatasetDict = adapter.load_subsets(
        lambda subset: adapter.load_subset(subset=subset, data_loader=CapturingDataLoader)
    )

    assert CapturingDataLoader.latest_limit is None
    assert CapturingDataLoader.latest_repeats == 1
    assert len(dataset_dict['subset-a']) == 6
    assert [sample.group_id for sample in dataset_dict['subset-a']] == [0, 0, 0, 1, 1, 1]
