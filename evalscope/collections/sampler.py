import random
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from tqdm import tqdm
from typing import List, Optional

from evalscope.collections.schema import CollectionSchema, DatasetInfo


@dataclass
class DatasetEntry:
    index: int = 0
    prompt: dict = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    task: str = ''
    weight: float = 0.0
    dataset_name: str = ''
    subset_name: str = ''


# Define an abstract base class for Samplers
class Sampler(ABC):

    def __init__(self, schema: CollectionSchema, count: Optional[int] = None):
        self.schema = schema
        self.count = count

    @abstractmethod
    def sample(self) -> List[dict]:
        raise NotImplementedError

    def _collect_dataset_data(self, dataset_info_list: List[DatasetInfo]) -> List[DatasetEntry]:
        all_data = []
        for dataset in tqdm(dataset_info_list, desc='Collecting dataset data'):
            data_dict = dataset.get_data()
            for subset_name, subset_data in data_dict.items():
                for prompt in subset_data:
                    all_data.append(
                        DatasetEntry(
                            prompt=prompt,
                            tags=dataset.tags,
                            task=dataset.task_type,
                            weight=dataset.weight,
                            dataset_name=dataset.name,
                            subset_name=subset_name,
                        ))
        return all_data

    def _update_index(self, all_data: List[DatasetEntry]) -> List[dict]:
        result = []
        for i, entry in enumerate(all_data):
            entry.index = i
            result.append(asdict(entry))
        return result


class WeightedSampler(Sampler):
    """
    Weighted sampler, according to the weight of each dataset, sample data from each dataset.
    """

    def sample(self) -> List[dict]:
        dataset_info_list = self.schema.flatten()
        all_data = self._collect_dataset_data(dataset_info_list)

        remaining_count = self.count
        sampled_data = []

        for i, dataset in enumerate(tqdm(dataset_info_list, desc='Sampling data')):
            if i == len(dataset_info_list) - 1:
                dataset_sample_count = remaining_count
            else:
                dataset_sample_count = int(dataset.weight * self.count)
                remaining_count -= dataset_sample_count

            sampled_data.extend(random.choices(all_data, k=dataset_sample_count))

        return self._update_index(sampled_data)


class UniformSampler(Sampler):
    """
    Uniform sampler, sample data from each dataset with the same number of samples.
    """

    def sample(self) -> List[dict]:
        dataset_info_list = self.schema.flatten()
        all_data = self._collect_dataset_data(dataset_info_list)

        num_datasets = len(dataset_info_list)
        samples_per_dataset = self.count // num_datasets
        sampled_data = []

        for _ in tqdm(dataset_info_list, desc='Sampling data'):
            sampled_data.extend(random.choices(all_data, k=samples_per_dataset))

        return self._update_index(sampled_data)


class StratifiedSampler(Sampler):
    """
    Stratified sampler, sample data from each dataset according to the number of samples of each dataset.
    """

    def sample(self) -> List[dict]:
        dataset_info_list = self.schema.flatten()
        all_data = self._collect_dataset_data(dataset_info_list)

        total_samples = sum(len(dataset.get_data()) for dataset in dataset_info_list)
        sampled_data = []

        for dataset in tqdm(dataset_info_list, desc='Sampling data'):
            dataset_samples = len(dataset.get_data())
            samples_for_dataset = int((dataset_samples / total_samples) * self.count)
            sampled_data.extend(random.choices(all_data, k=samples_for_dataset))

        return self._update_index(sampled_data)


if __name__ == '__main__':
    from evalscope.utils.io_utils import dump_jsonl_data

    schema = CollectionSchema.from_json('outputs/schema.json')
    print(schema.to_dict())
    mixed_data = WeightedSampler(schema, 100).sample()
    dump_jsonl_data(mixed_data, 'outputs/weighted_mixed_data.jsonl')

    mixed_data = UniformSampler(schema, 100).sample()
    dump_jsonl_data(mixed_data, 'outputs/uniform_mixed_data.jsonl')

    mixed_data = StratifiedSampler(schema, 100).sample()
    dump_jsonl_data(mixed_data, 'outputs/stratified_mixed_data.jsonl')
