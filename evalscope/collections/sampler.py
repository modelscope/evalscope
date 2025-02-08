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
    categories: List[str] = field(default_factory=list)
    task_type: str = ''
    weight: float = 0.0
    dataset_name: str = ''
    subset_name: str = ''


# Define an abstract base class for Samplers
class Sampler(ABC):

    def __init__(self, schema: CollectionSchema):
        self.schema = schema

    @abstractmethod
    def sample(self) -> List[dict]:
        raise NotImplementedError

    def _sample_dataset(self, dataset: DatasetInfo, count: int) -> List[DatasetEntry]:
        all_data = []
        data_dict = dataset.get_data()
        for subset_name, subset_data in data_dict.items():
            for prompt in subset_data:
                all_data.append(
                    DatasetEntry(
                        prompt=prompt,
                        tags=dataset.tags,
                        categories=dataset.hierarchy,
                        task_type=dataset.task_type,
                        weight=dataset.weight,
                        dataset_name=dataset.name,
                        subset_name=subset_name,
                    ))
        count = min(count, len(all_data))  # avoid sampling more than the dataset size
        sampled_data = random.sample(all_data, k=count)
        return sampled_data

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

    def sample(self, count: int) -> List[dict]:
        dataset_info_list = self.schema.flatten()
        sampled_data = []
        remaining_count = count

        for i, dataset in enumerate(tqdm(dataset_info_list, desc='Sampling data')):
            if i == len(dataset_info_list) - 1:
                dataset_sample_count = remaining_count
            else:
                dataset_sample_count = int(dataset.weight * count)
                remaining_count -= dataset_sample_count

            sampled_data.extend(self._sample_dataset(dataset, dataset_sample_count))

        return self._update_index(sampled_data)


class UniformSampler(Sampler):
    """
    Uniform sampler, sample data from each dataset with the same number of samples.
    """

    def sample(self, count: int) -> List[dict]:
        dataset_info_list = self.schema.flatten()
        num_datasets = len(dataset_info_list)
        remaining_count = count
        sampled_data = []

        for i, dataset in enumerate(tqdm(dataset_info_list, desc='Sampling data')):
            if i == len(dataset_info_list) - 1:
                dataset_sample_count = remaining_count
            else:
                dataset_sample_count = count // num_datasets
                remaining_count -= dataset_sample_count

            sampled_data.extend(self._sample_dataset(dataset, dataset_sample_count))

        return self._update_index(sampled_data)


class StratifiedSampler(Sampler):
    """
    Stratified sampler, sample data from each dataset according to the number of samples of each dataset.
    """

    def sample(self, count: int) -> List[dict]:
        dataset_info_list = self.schema.flatten()

        total_samples = sum(len(dataset.get_data()) for dataset in dataset_info_list)
        remaining_count = count
        sampled_data = []

        for i, dataset in enumerate(tqdm(dataset_info_list, desc='Sampling data')):
            if i == len(dataset_info_list) - 1:
                dataset_sample_count = remaining_count
            else:
                dataset_sample_count = int((len(dataset.get_data()) / total_samples) * count)
                remaining_count -= dataset_sample_count

            sampled_data.extend(self._sample_dataset(dataset, dataset_sample_count))
        return self._update_index(sampled_data)


if __name__ == '__main__':
    from evalscope.utils.io_utils import dump_jsonl_data

    schema = CollectionSchema.from_json('outputs/schema.json')
    print(schema.to_dict())
    mixed_data = WeightedSampler(schema).sample(10)
    dump_jsonl_data(mixed_data, 'outputs/weighted_mixed_data.jsonl')

    # mixed_data = UniformSampler(schema, 100).sample()
    # dump_jsonl_data(mixed_data, 'outputs/uniform_mixed_data.jsonl')

    # mixed_data = StratifiedSampler(schema, 100).sample()
    # dump_jsonl_data(mixed_data, 'outputs/stratified_mixed_data.jsonl')
