from dataclasses import dataclass, field
from typing import Callable


@dataclass
class Metric:
    name: str = 'default_metric'
    object: Callable = field(default_factory=lambda: None)
