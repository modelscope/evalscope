from dataclasses import asdict, dataclass
from functools import wraps
from typing import Dict, List, Optional, Union

from evalscope.constants import EvalType
from evalscope.utils.filters import Filter


@dataclass
class PromptData:
    data: List[str]
    index: Optional[Union[int, str]] = 0
    system_prompt: Optional[str] = None
    multi_choices: Optional[List[str]] = None
    id: Optional[str] = None
    messages: Optional[List[dict]] = None

    def to_dict(self) -> Dict:
        return {k: v for k, v in asdict(self).items() if v is not None}


def preprocess_decorator(func):

    @wraps(func)
    def wrapper(self, result: str, raw_input_d: dict = None, **kwargs):
        if result is None:
            result = ''
        filters = self.config_kwargs.get('filters', None)
        if filters:
            # Apply filters to the resultply filters to the result
            for filter_name, filter_value in filters.items():
                result = Filter.apply(filter_name, result, filter_value)
        return func(self, result, raw_input_d, **kwargs)

    return wrapper


def load_file_with_extension(file_path: Union[str, List[str]]) -> List[dict]:
    """
    Load a file with a specific extension and return its content as a list of dictionaries.
    """
    import json
    import os

    if isinstance(file_path, str):
        file_path = [file_path]

    data = []
    for path in file_path:
        if not os.path.exists(path):
            raise FileNotFoundError(f'The file {path} does not exist.')

        with open(path, 'r', encoding='utf-8') as f:
            if path.endswith('.json'):
                data.extend(json.load(f))
            elif path.endswith('.jsonl'):
                data.extend([json.loads(line) for line in f])
            elif path.endswith('.txt'):
                data.extend([{'text': f.read()}])
    return data
