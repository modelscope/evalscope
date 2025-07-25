import copy
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Optional, Type, Union

from evalscope.constants import OutputType

if TYPE_CHECKING:
    from evalscope.api.benchmark import DataAdapter


@dataclass
class BenchmarkMeta:
    """Metadata for a benchmark, including dataset and model configurations."""
    name: str
    dataset_id: str
    data_adapter: Type['DataAdapter']
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
    tags: Optional[List[str]] = field(default_factory=list)
    filters: Optional[OrderedDict] = None
    extra_params: Optional[Dict] = field(default_factory=dict)

    def _update(self, args: dict):
        if args.get('local_path'):
            self.dataset_id = args['local_path']
            del args['local_path']
        self.__dict__.update(args)

    def to_dict(self) -> dict:
        return self.__dict__.copy()

    def to_string_dict(self) -> dict:
        cur_dict = copy.deepcopy(self.to_dict())
        # cur_dict['data_adapter'] = self.data_adapter.__name__
        del cur_dict['data_adapter']
        return cur_dict
