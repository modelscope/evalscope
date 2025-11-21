import copy
import json
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Union

from evalscope.api.dataset import DatasetDict
from evalscope.api.registry import get_benchmark
from evalscope.config import TaskConfig


@dataclass
class DatasetInfo:
    """Metadata and configuration for a single dataset in a collection."""
    name: str
    weight: float = 1.0  # dataset-level weight in the collection
    task_type: str = ''
    tags: List[str] = field(default_factory=list)
    args: Dict[str, object] = field(default_factory=dict)
    hierarchy: List[str] = field(default_factory=list)

    def get_data(self) -> DatasetDict:
        """Load dataset data using the benchmark registry."""
        dataset_args = {self.name: self.args}
        benchmark_meta = get_benchmark(self.name, config=TaskConfig(dataset_args=dataset_args))
        data_dict = benchmark_meta.load_dataset()
        return data_dict


def flatten_weight(collection: 'CollectionSchema', base_weight: float = 1.0) -> None:
    """Propagate and normalize dataset weights within a nested collection."""
    total_weight = sum(dataset.weight for dataset in collection.datasets)
    for dataset in collection.datasets:
        current_weight = dataset.weight / total_weight * base_weight
        if isinstance(dataset, CollectionSchema):
            flatten_weight(dataset, current_weight)
        else:
            dataset.weight = current_weight


def flatten_name(collection: 'CollectionSchema', parent_names: Optional[List[str]] = None) -> None:
    """Populate hierarchy names for datasets based on nested collection names."""
    if parent_names is None:
        parent_names = []
    current_names = parent_names + [collection.name]
    for dataset in collection.datasets:
        if isinstance(dataset, CollectionSchema):
            flatten_name(dataset, current_names)
        else:
            dataset.hierarchy = current_names.copy()


def flatten_datasets(collection: 'CollectionSchema') -> List[DatasetInfo]:
    """Flatten nested collections into a flat list of DatasetInfo."""
    flat_datasets: List[DatasetInfo] = []
    for dataset in collection.datasets:
        if isinstance(dataset, CollectionSchema):
            flat_datasets.extend(flatten_datasets(dataset))
        else:
            flat_datasets.append(dataset)
    return flat_datasets


@dataclass
class CollectionSchema:
    """Schema describing a collection of datasets, possibly nested."""
    name: str
    weight: float = 1.0
    datasets: List[Union[DatasetInfo, 'CollectionSchema']] = field(default_factory=list)

    def __str__(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=4)

    def to_dict(self) -> Dict[str, object]:
        return {
            'name':
            self.name,
            'weight':
            self.weight,
            'datasets':
            [asdict(dataset) if isinstance(dataset, DatasetInfo) else dataset.to_dict() for dataset in self.datasets],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, object]) -> 'CollectionSchema':
        instance = cls(name=data.get('name', ''), weight=data.get('weight', 1))  # type: ignore[arg-type]
        for dataset in data.get('datasets', []):  # type: ignore[assignment]
            if 'datasets' in dataset:
                instance.datasets.append(CollectionSchema.from_dict(dataset))
            else:
                instance.datasets.append(DatasetInfo(**dataset))
        return instance

    def dump_json(self, file_path: str) -> None:
        d = self.to_dict()
        with open(file_path, 'w') as f:
            json.dump(d, f, ensure_ascii=False, indent=4)

    @classmethod
    def from_json(cls, file_path: str) -> 'CollectionSchema':
        with open(file_path, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)

    def flatten(self) -> List[DatasetInfo]:
        """Return a flat list of DatasetInfo with propagated weights and hierarchy."""
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
                datasets=[DatasetInfo(name='ceval', weight=1, tags=['zh'], args={'subset_list': ['logic']})]
            )
        ]
    )
    print(schema)
    print(schema.flatten())
    schema.dump_json('outputs/schema.json')

    schema = CollectionSchema.from_json('outputs/schema.json')
    print(schema)
    # 打印扁平化后的结果
    for dataset in schema.flatten():
        print(f'Dataset: {dataset.name}')
        print(f"Hierarchy: {' -> '.join(dataset.hierarchy)}")
