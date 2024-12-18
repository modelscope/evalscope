import copy
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from evalscope.benchmarks import DataAdapter
from evalscope.models import BaseModelAdapter

BENCHMARK_MAPPINGS = {}


@dataclass
class BenchmarkMeta:
    name: str
    dataset_id: str
    data_adapter: DataAdapter
    model_adapter: BaseModelAdapter
    subset_list: List[str] = field(default_factory=list)
    metric_list: List[Dict] = field(default_factory=list)
    few_shot_num: int = 0
    few_shot_random: bool = False
    train_split: Optional[str] = None
    eval_split: Optional[str] = None
    prompt_template: str = ''

    def update(self, args: dict):
        if args.get('local_path'):
            self.dataset_id = args['local_path']
            del args['local_path']
        self.__dict__.update(args)

    def to_dict(self):
        cur_dict = copy.deepcopy(self.__dict__)
        del cur_dict['data_adapter']
        del cur_dict['model_adapter']
        return cur_dict


class Benchmark:

    def __init__(self):
        pass

    @classmethod
    def get(cls, name: str) -> 'BenchmarkMeta':
        if name not in BENCHMARK_MAPPINGS:
            raise Exception(f"Unknown benchmark: {name}. Available tasks: {BENCHMARK_MAPPINGS.keys()}")
        benchmark = BENCHMARK_MAPPINGS[name]
        return benchmark

    @classmethod
    def register(cls, name: str, dataset_id: str, model_adapter: BaseModelAdapter, **kwargs):

        def register_wrapper(data_adapter):
            if name in BENCHMARK_MAPPINGS:
                raise Exception(f"Benchmark {name} already registered")
            BENCHMARK_MAPPINGS[name] = BenchmarkMeta(
                name=name, data_adapter=data_adapter, model_adapter=model_adapter, dataset_id=dataset_id, **kwargs)
            return data_adapter

        return register_wrapper
