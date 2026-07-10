import operator
import re
from dataclasses import dataclass
from typing import Callable

from evalscope.perf.domain.errors import PerfConfigError
from evalscope.perf.domain.result import RunResult


@dataclass(frozen=True)
class Criterion:
    operation: Callable[[float, float], bool]
    threshold: float

    def validate(self, value: float) -> bool:
        return self.operation(value, self.threshold)


_OPERATORS = {
    '<': operator.lt,
    '<=': operator.le,
    '>': operator.gt,
    '>=': operator.ge,
}


def parse_criterion(value: str) -> Criterion:
    match = re.fullmatch(r'\s*(<=|>=|<|>)\s*(-?\d+(?:\.\d+)?)\s*', value)
    if not match:
        raise PerfConfigError(f'Invalid SLA criterion {value!r}; expected an operator and number')
    operation, threshold = match.groups()
    return Criterion(_OPERATORS[operation], float(threshold))


def metric_value(result: RunResult, name: str) -> float:
    summary = result.summary
    values = {
        'avg_latency': summary.averages.get('latency', 0),
        'avg_ttft': summary.averages.get('ttft', 0),
        'avg_tpot': summary.averages.get('tpot', 0),
        'p99_latency': _p99(result, 'latency'),
        'p99_ttft': _p99(result, 'ttft'),
        'p99_tpot': _p99(result, 'tpot'),
        'rps': summary.request_throughput,
        'output_tps': summary.output_token_throughput or 0,
        'success_rate': summary.success_rate,
    }
    if name not in values:
        raise PerfConfigError(f'Unsupported SLA metric {name!r}. Available metrics: {", ".join(sorted(values))}')
    return float(values[name])


def objective_value(result: RunResult, objective: str) -> float:
    if objective == 'max_rps':
        return result.summary.request_throughput
    if objective == 'max_output_tps':
        return result.summary.output_token_throughput or 0
    if objective == 'min_latency':
        return result.summary.averages.get('latency', float('inf'))
    raise PerfConfigError(f'Unsupported SLA objective {objective!r}')


def _p99(result: RunResult, name: str) -> float:
    stats = result.percentiles.get(name)
    return float(stats.p99 or 0) if stats else 0.0
