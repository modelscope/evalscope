import json
import random
from abc import ABC, abstractmethod
from typing import List, Optional

from evalscope.collections.collection_schema import CollectionSchema


# Define an abstract base class for Samplers
class Sampler(ABC):

    def __init__(self, schema: CollectionSchema, count: Optional[int] = None):
        self.schema = schema
        self.count = count

    @abstractmethod
    def sample(self) -> List[dict]:
        pass


class WeightedSampler(Sampler):

    def sample(self) -> List[dict]:
        all_data = []

        dataset_info_list = self.schema.flatten()
        total_weight = sum(dataset.weight for dataset in dataset_info_list)

        remaining_count = self.count

        for i, dataset in enumerate(dataset_info_list):
            data_dict = dataset.get_data()

            dataset_data = []
            for subset_name, subset_data in data_dict.items():
                for prompt in subset_data:
                    dataset_data.append({
                        'prompt': prompt,
                        'tags': dataset.tags,
                        'task': dataset.task_type,
                        'source': f'{dataset.name}/{subset_name}',
                    })

            # For the last dataset, use the remaining count
            if i == len(dataset_info_list) - 1:
                dataset_sample_count = remaining_count
            else:
                dataset_sample_count = int((dataset.weight / total_weight) * self.count)
                remaining_count -= dataset_sample_count

            sampled_data = random.choices(dataset_data, k=dataset_sample_count)
            all_data.extend(sampled_data)

        return all_data


def save_to_jsonl(data, file_path):
    with open(file_path, 'w') as f:
        for i, entry in enumerate(data):
            entry['id'] = i
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')


if __name__ == '__main__':
    schema = CollectionSchema.from_dict(json.load(open('schema.json', 'r')))
    print(schema.to_dict())
    mixed_data = WeightedSampler(schema, 10).sample()
    save_to_jsonl(mixed_data, 'outputs/mixed_data.jsonl')
