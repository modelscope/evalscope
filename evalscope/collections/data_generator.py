import json
import random
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from tqdm import tqdm
from typing import List, Optional

from evalscope.collections.schema import CollectionSchema
from evalscope.utils.io_utils import dump_jsonl_data


# Define an abstract base class for Samplers
class Sampler(ABC):

    def __init__(self, schema: CollectionSchema, count: Optional[int] = None):
        self.schema = schema
        self.count = count

    @abstractmethod
    def sample(self) -> List[dict]:
        pass


@dataclass
class DatasetEntry:
    index: int = 0
    prompt: dict = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    task: str = ''
    weight: float = 0.0
    dataset_name: str = ''
    subset_name: str = ''


class WeightedSampler(Sampler):

    def sample(self) -> List[dict]:
        all_data: List[DatasetEntry] = []

        dataset_info_list = self.schema.flatten()
        total_weight = sum(dataset.weight for dataset in dataset_info_list)

        remaining_count = self.count

        for i, dataset in enumerate(tqdm(dataset_info_list, desc='Sampling data')):
            data_dict = dataset.get_data()

            dataset_data = []
            for subset_name, subset_data in data_dict.items():
                for prompt in subset_data:
                    dataset_data.append(
                        DatasetEntry(
                            prompt=prompt,
                            tags=dataset.tags,
                            task=dataset.task_type,
                            weight=dataset.weight,
                            dataset_name=dataset.name,
                            subset_name=subset_name,
                        ))

            # For the last dataset, use the remaining count
            if i == len(dataset_info_list) - 1:
                dataset_sample_count = remaining_count
            else:
                dataset_sample_count = int((dataset.weight / total_weight) * self.count)
                remaining_count -= dataset_sample_count

            sampled_data = random.choices(dataset_data, k=dataset_sample_count)
            all_data.extend(sampled_data)

        # update index
        result = []
        for i, entry in enumerate(all_data):
            entry.index = i
            result.append(asdict(entry))
        return result


if __name__ == '__main__':
    schema = CollectionSchema.from_dict(json.load(open('outputs/schema.json', 'r')))
    print(schema.to_dict())
    mixed_data = WeightedSampler(schema, 10).sample()
    dump_jsonl_data(mixed_data, 'outputs/mixed_data.jsonl')
