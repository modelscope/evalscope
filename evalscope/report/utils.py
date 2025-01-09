import json
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List

from evalscope.metrics import macro_mean, micro_mean
from evalscope.utils import normalize_score


@dataclass
class Subset:
    name: str = 'default_subset'
    score: float = 0.0
    num: int = 0

    def __post_init__(self):
        self.score = normalize_score(self.score)


@dataclass
class Category:
    name: str = 'default_category'
    num: int = 0
    score: float = 0.0
    macro_score: float = 0.0
    subsets: List[Subset] = field(default_factory=list)

    def __post_init__(self):
        self.num = sum(subset.num for subset in self.subsets)
        self.score = normalize_score(micro_mean(self.subsets))
        self.macro_score = normalize_score(macro_mean(self.subsets))

    @classmethod
    def from_dict(cls, data: dict):
        subsets = [Subset(**subset) for subset in data.get('subsets', [])]
        return cls(name=data['name'], subsets=subsets)


@dataclass
class Metric:
    name: str = 'default_metric'
    num: int = 0
    score: float = 0.0
    macro_score: float = 0.0
    categories: List[Category] = field(default_factory=list)

    def __post_init__(self):
        self.num = sum(category.num for category in self.categories)
        self.score = normalize_score(micro_mean(self.categories))
        self.macro_score = normalize_score(macro_mean(self.categories))

    @classmethod
    def from_dict(cls, data: dict):
        categories = [Category.from_dict(category) for category in data.get('categories', [])]
        return cls(name=data['name'], categories=categories)


@dataclass
class Report:
    name: str = 'default_report'
    dataset_name: str = 'default_dataset'
    model_name: str = 'default_model'
    score: float = 0.0
    metrics: List[Metric] = field(default_factory=list)

    def __post_init__(self):
        self.score = normalize_score(macro_mean(self.metrics))

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict):
        metrics = [Metric.from_dict(metric) for metric in data.get('metrics', [])]
        return cls(
            name=data['name'],
            score=data['score'],
            metrics=metrics,
            dataset_name=data['dataset_name'],
            model_name=data['model_name'])

    @classmethod
    def from_json(cls, json_file: str):
        with open(json_file, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)
