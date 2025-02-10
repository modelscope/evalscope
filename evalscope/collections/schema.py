import copy
import json
from dataclasses import asdict, dataclass, field
from typing import List, Union


@dataclass
class DatasetInfo:
    name: str
    weight: float = 1.0  # sample weight in each collection
    task_type: str = ''
    tags: List[str] = field(default_factory=list)
    args: dict = field(default_factory=dict)
    hierarchy: List[str] = field(default_factory=list)

    def get_data(self) -> dict:
        from evalscope.benchmarks import Benchmark

        benchmark_meta = Benchmark.get(self.name)

        data_adapter = benchmark_meta.get_data_adapter(config=self.args)
        data_dict = data_adapter.load()
        prompts = data_adapter.gen_prompts(data_dict)
        return prompts


def flatten_weight(collection: 'CollectionSchema', base_weight=1):
    total_weight = sum(dataset.weight for dataset in collection.datasets)
    for dataset in collection.datasets:
        current_weight = dataset.weight / total_weight * base_weight
        if isinstance(dataset, CollectionSchema):
            flatten_weight(dataset, current_weight)
        else:
            dataset.weight = current_weight


def flatten_name(collection: 'CollectionSchema', parent_names=None):
    if parent_names is None:
        parent_names = []
    current_names = parent_names + [collection.name]
    for dataset in collection.datasets:
        if isinstance(dataset, CollectionSchema):
            flatten_name(dataset, current_names)
        else:
            dataset.hierarchy = current_names.copy()


def flatten_datasets(collection: 'CollectionSchema') -> List[DatasetInfo]:
    flat_datasets = []
    for dataset in collection.datasets:
        if isinstance(dataset, CollectionSchema):
            flat_datasets.extend(flatten_datasets(dataset))
        else:
            flat_datasets.append(dataset)
    return flat_datasets


@dataclass
class CollectionSchema:
    name: str
    weight: float = 1.0
    datasets: List[Union[DatasetInfo, 'CollectionSchema']] = field(default_factory=list)

    def __str__(self):
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=4)

    def to_dict(self):
        return {
            'name':
            self.name,
            'weight':
            self.weight,
            'datasets':
            [asdict(dataset) if isinstance(dataset, DatasetInfo) else dataset.to_dict() for dataset in self.datasets],
        }

    @classmethod
    def from_dict(cls, data):
        instance = cls(name=data.get('name', ''), weight=data.get('weight', 1))
        for dataset in data.get('datasets', []):
            if 'datasets' in dataset:
                instance.datasets.append(CollectionSchema.from_dict(dataset))
            else:
                instance.datasets.append(DatasetInfo(**dataset))
        return instance

    def dump_json(self, file_path):
        d = self.to_dict()
        with open(file_path, 'w') as f:
            json.dump(d, f, ensure_ascii=False, indent=4)

    @classmethod
    def from_json(cls, file_path):
        with open(file_path, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)

    def flatten(self) -> List[DatasetInfo]:
        collection = copy.deepcopy(self)
        flatten_name(collection)
        flatten_weight(collection)
        return flatten_datasets(collection)


if __name__ == '__main__':
    schema = CollectionSchema(
        name='reasoning',
        datasets=[
            CollectionSchema(name='english', datasets=[
                DatasetInfo(name='arc', weight=1, tags=['en']),
            ]),
            CollectionSchema(
                name='chinese',
                datasets=[DatasetInfo(name='ceval', weight=1, tags=['zh'], args={'subset_list': ['logic']})])
        ])
    print(schema)
    print(schema.flatten())
    schema.dump_json('outputs/schema.json')

    schema = CollectionSchema.from_json('outputs/schema.json')
    print(schema)
    # 打印扁平化后的结果
    for dataset in schema.flatten():
        print(f'Dataset: {dataset.name}')
        print(f"Hierarchy: {' -> '.join(dataset.hierarchy)}")
