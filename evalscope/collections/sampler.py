import random
from abc import ABC, abstractmethod
from pydantic import BaseModel, Field
from tqdm import tqdm
from typing import List

from evalscope.collections.schema import CollectionSchema, DatasetInfo


class DatasetEntry(BaseModel):
    index: int = 0
    prompt: dict = Field(default_factory=dict)
    tags: List[str] = Field(default_factory=list)
    categories: List[str] = Field(default_factory=list)
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
        all_data: List[DatasetEntry] = []
        data_dict = dataset.get_data()

        # Compute overall count (total items across all subsets)
        total_count = min(count, sum(len(subset_data) for subset_data in data_dict.values()))
        if total_count == 0:
            return []

        for subset_name, subset_data in data_dict.items():
            for sample in subset_data:
                all_data.append(
                    DatasetEntry(
                        prompt=sample.model_dump(exclude_none=True),
                        tags=dataset.tags,
                        categories=dataset.hierarchy,
                        task_type=dataset.task_type,
                        weight=dataset.weight / float(total_count),
                        dataset_name=dataset.name,
                        subset_name=subset_name,
                    )
                )

        sampled_data = random.sample(all_data, k=total_count)
        return sampled_data

    def _update_index(self, all_data: List[DatasetEntry]) -> List[dict]:
        result = []
        for i, entry in enumerate(all_data):
            entry.index = i
            result.append(entry.model_dump())
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

        # Precompute sample counts per dataset once to avoid repeated get_data() calls
        per_dataset_counts = [
            sum(len(subset_data) for subset_data in dataset.get_data().values()) for dataset in dataset_info_list
        ]
        total_samples = sum(per_dataset_counts)

        remaining_count = count
        sampled_data = []

        for i, dataset in enumerate(tqdm(dataset_info_list, desc='Sampling data')):
            if i == len(dataset_info_list) - 1:
                dataset_sample_count = remaining_count
            else:
                ds_total = per_dataset_counts[i]
                dataset_sample_count = int((ds_total / total_samples) * count) if total_samples > 0 else 0
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
