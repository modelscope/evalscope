from dataclasses import dataclass
from functools import wraps
from typing import Dict, List, Optional

from evalscope.constants import EvalType
from evalscope.utils.filters import Filter


@dataclass
class PromptData:
    data: List[str]
    index: Optional[int] = 0
    system_prompt: Optional[str] = None
    multi_choices: Optional[List[str]] = None

    def to_dict(self) -> Dict:
        if self.multi_choices is None:
            return {
                'data': self.data,
                'index': self.index,
                'system_prompt': self.system_prompt,
            }
        else:
            return {
                'data': self.data,
                'index': self.index,
                'system_prompt': self.system_prompt,
                'multi_choices': self.multi_choices,
            }


def preprocess_decorator(func):

    @wraps(func)
    def wrapper(self, result: str, raw_input_d: dict = None, eval_type: str = EvalType.CHECKPOINT):
        filters = self.config_kwargs.get('filters', None)
        if filters:
            # Apply filters to the resultply filters to the result
            for filter_name, filter_value in filters.items():
                result = Filter.apply(filter_name, result, filter_value)
        return func(self, result, raw_input_d, eval_type)

    return wrapper
