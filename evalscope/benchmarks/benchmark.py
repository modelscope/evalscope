import copy
from collections import OrderedDict
from dataclasses import dataclass, field, fields
from typing import TYPE_CHECKING, Dict, List, Optional

from evalscope.constants import OutputType

if TYPE_CHECKING:
    from evalscope.benchmarks import DataAdapter

BENCHMARK_MAPPINGS = {}


@dataclass
class BenchmarkMeta:
    name: str
    dataset_id: str
    data_adapter: 'DataAdapter'
    model_adapter: Optional[str] = OutputType.GENERATION
    output_types: Optional[List[str]] = field(default_factory=lambda: [OutputType.GENERATION])
    subset_list: List[str] = field(default_factory=lambda: ['default'])
    metric_list: List[str] = field(default_factory=list)
    few_shot_num: int = 0
    few_shot_random: bool = False
    train_split: Optional[str] = None
    eval_split: Optional[str] = None
    prompt_template: Optional[str] = None
    system_prompt: Optional[str] = None
    query_template: Optional[str] = None
    pretty_name: Optional[str] = None
    description: Optional[str] = None
    filters: Optional[OrderedDict] = None
    extra_params: Optional[Dict] = field(default_factory=dict)

    def _update(self, args: dict):
        if args.get('local_path'):
            self.dataset_id = args['local_path']
            del args['local_path']
        self.__dict__.update(args)

    def to_dict(self) -> dict:
        return self.__dict__

    def to_string_dict(self) -> dict:
        cur_dict = copy.deepcopy(self.to_dict())
        # cur_dict['data_adapter'] = self.data_adapter.__name__
        del cur_dict['data_adapter']
        return cur_dict

    def get_data_adapter(self, config: dict = {}) -> 'DataAdapter':
        if config:
            self._update(config)

        data_adapter = self.data_adapter(**self.to_dict())
        return data_adapter


class Benchmark:

    def __init__(self):
        pass

    @classmethod
    def get(cls, name: str) -> 'BenchmarkMeta':
        if name not in BENCHMARK_MAPPINGS:
            raise Exception(f'Unknown benchmark: {name}. Available tasks: {list(BENCHMARK_MAPPINGS.keys())}')
        benchmark = BENCHMARK_MAPPINGS[name]
        return benchmark

    @classmethod
    def register(cls, name: str, dataset_id: str, **kwargs):

        def register_wrapper(data_adapter):
            if name in BENCHMARK_MAPPINGS:
                raise Exception(f'Benchmark {name} already registered')
            BENCHMARK_MAPPINGS[name] = BenchmarkMeta(
                name=name, data_adapter=data_adapter, dataset_id=dataset_id, **kwargs)
            return data_adapter

        return register_wrapper
