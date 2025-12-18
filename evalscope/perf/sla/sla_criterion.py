from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class SLACriterionBase(ABC):
    target: float

    @abstractmethod
    def validate(self, actual: float) -> bool:
        raise NotImplementedError

    @abstractmethod
    def format_cond(self, lhs: str) -> str:
        raise NotImplementedError


@dataclass
class SLALessThan(SLACriterionBase):

    def validate(self, actual: float) -> bool:
        return actual < self.target

    def format_cond(self, lhs: str) -> str:
        return f'{lhs} < {self.target}'

    def __str__(self):
        return f'< {self.target}'


@dataclass
class SLALessThanOrEqualTo(SLACriterionBase):

    def validate(self, actual: float) -> bool:
        return actual <= self.target

    def format_cond(self, lhs: str) -> str:
        return f'{lhs} <= {self.target}'

    def __str__(self):
        return f'<= {self.target}'


@dataclass
class SLAGreaterThan(SLACriterionBase):

    def validate(self, actual: float) -> bool:
        return actual > self.target

    def format_cond(self, lhs: str) -> str:
        return f'{lhs} > {self.target}'

    def __str__(self):
        return f'> {self.target}'


@dataclass
class SLAGreaterThanOrEqualTo(SLACriterionBase):

    def validate(self, actual: float) -> bool:
        return actual >= self.target

    def format_cond(self, lhs: str) -> str:
        return f'{lhs} >= {self.target}'

    def __str__(self):
        return f'>= {self.target}'


@dataclass
class SLAMax(SLACriterionBase):

    def validate(self, actual: float) -> bool:
        return True

    def format_cond(self, lhs: str) -> str:
        return f'{lhs} -> max'

    def __str__(self):
        return 'max'


@dataclass
class SLAMin(SLACriterionBase):

    def validate(self, actual: float) -> bool:
        return True

    def format_cond(self, lhs: str) -> str:
        return f'{lhs} -> min'

    def __str__(self):
        return 'min'


SLA_CRITERIA = {
    '<=': SLALessThanOrEqualTo,
    '>=': SLAGreaterThanOrEqualTo,
    '<': SLALessThan,
    '>': SLAGreaterThan,
}


def create_criterion(value_str: str) -> SLACriterionBase:
    value_str = str(value_str).strip()
    if value_str == 'max':
        return SLAMax(target=0.0)
    if value_str == 'min':
        return SLAMin(target=0.0)

    for op_key in sorted(SLA_CRITERIA.keys(), key=len, reverse=True):
        if value_str.startswith(op_key):
            try:
                val = float(value_str[len(op_key):])
                return SLA_CRITERIA[op_key](val)
            except ValueError:
                raise ValueError(f'Invalid target value in SLA param: {value_str}')
