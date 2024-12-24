import json
from dataclasses import asdict, dataclass, field
from typing import List, Union

from evalscope.benchmarks.benchmark import Benchmark


@dataclass
class DatasetInfo:
    name: str
    weight: int = 1  # sample weight in each collection
    task_type: str = ''
    tags: List[str] = field(default_factory=list)
    args: dict = field(default_factory=dict)

    def get_data(self) -> dict:
        benchmark_meta = Benchmark.get(self.name)

        data_adapter = benchmark_meta.get_data_adapter(config=self.args)
        data_dict = data_adapter.load(
            dataset_name_or_path=benchmark_meta.dataset_id, subset_list=benchmark_meta.subset_list)
        prompts = data_adapter.gen_prompts(data_dict)
        return prompts


@dataclass
class CollectionSchema:
    name: str
    datasets: List[Union[DatasetInfo, 'CollectionSchema']] = field(default_factory=list)

    def __post_init__(self):
        # uniform the weight of datasets in each collection
        total_weight = sum(dataset.weight for dataset in self.datasets if isinstance(dataset, DatasetInfo))
        for dataset in self.datasets:
            if isinstance(dataset, DatasetInfo):
                dataset.weight = dataset.weight / total_weight

    def add_dataset(self, name, weight=1, task_type='', tags=[]):
        self.datasets.append(DatasetInfo(name, weight, task_type, tags))

    def add_collection(self, collection: 'CollectionSchema'):
        self.datasets.append(collection)

    def get_datasets(self):
        return self.datasets

    def to_dict(self):
        return {
            'name':
            self.name,
            'datasets':
            [asdict(dataset) if isinstance(dataset, DatasetInfo) else dataset.to_dict() for dataset in self.datasets]
        }

    @classmethod
    def from_dict(cls, data):
        instance = cls(name=data.get('name', ''))
        for dataset in data.get('datasets', []):
            if 'datasets' in dataset:
                instance.datasets.append(CollectionSchema.from_dict(dataset))
            else:
                instance.datasets.append(DatasetInfo(**dataset))
        return instance

    def flatten(self) -> List[DatasetInfo]:
        flat_datasets = []

        for dataset in self.datasets:
            if isinstance(dataset, CollectionSchema):
                nested_datasets = dataset.flatten()
                flat_datasets.extend(nested_datasets)
            else:
                flat_datasets.append(dataset)
        return flat_datasets

    def dump_json(self, file_path):
        d = self.to_dict()
        with open(file_path, 'w') as f:
            json.dump(d, f, ensure_ascii=False, indent=4)


if __name__ == '__main__':
    schema = CollectionSchema(
        name='math&reasoning',
        datasets=[
            CollectionSchema(
                name='math',
                datasets=[
                    DatasetInfo(name='gsm8k', weight=1, task_type='math', tags=['en', 'math']),
                    DatasetInfo(name='competition_math', weight=2, task_type='math', tags=['en', 'math']),
                ]),
            CollectionSchema(
                name='reasoning',
                datasets=[
                    DatasetInfo(name='arc', weight=1, task_type='reasoning', tags=['en', 'reasoning']),
                ]),
        ])
    print(schema.to_dict())
    print(schema.flatten())
    schema.dump_json('outputs/schema.json')

    schema = CollectionSchema.from_dict(json.load(open('outputs/schema.json', 'r')))
    print(schema.to_dict())
    print(schema.flatten())
