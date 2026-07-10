from __future__ import annotations

import math
from statistics import median
from typing import Dict, List, Optional, Tuple

from evalscope.perf.config.models import ClosedLoopLoad, OpenLoopLoad, PerfConfig
from evalscope.perf.config.resolve import ResolvedRunSpec
from evalscope.perf.domain.errors import PerfConfigError
from evalscope.perf.domain.result import RunResult, SLAEvaluation, SLAResult
from evalscope.perf.engine.run_engine import RunEngine
from evalscope.perf.sla.criteria import metric_value, objective_value, parse_criterion


class SLATuner:
    """Noise-aware load search built directly on the asynchronous run engine."""

    def __init__(self, config: PerfConfig, run_id: str, suite_dir: str) -> None:
        if config.sla is None:
            raise PerfConfigError('SLA configuration is required')
        if len(config.suite.loads) != 1:
            raise PerfConfigError('SLA tuning requires exactly one template load')
        self.config = config
        self.sla = config.sla
        self.run_id = run_id
        self.suite_dir = suite_dir
        self.template = config.suite.loads[0]
        self._cache: Dict[str, Tuple[SLAEvaluation, List[RunResult]]] = {}
        self._validate_template()

    async def run(self) -> tuple[List[RunResult], SLAResult]:
        if self.sla.criteria:
            best = await self._constraint_search()
        else:
            best = await self._objective_search()
        records = sorted(self._cache.values(), key=lambda item: item[0].load_value)
        evaluations = [item[0] for item in records]
        runs = [run for _, items in records for run in items]
        return runs, SLAResult(variable=self.sla.variable, best_value=best, evaluations=evaluations)

    def _validate_template(self) -> None:
        if self.sla.variable == 'concurrency' and not isinstance(self.template, ClosedLoopLoad):
            raise PerfConfigError('concurrency SLA tuning requires a closed_loop template')
        if self.sla.variable == 'request_rate' and not isinstance(self.template, OpenLoopLoad):
            raise PerfConfigError('request_rate SLA tuning requires an open_loop template')

    def _load(self, value: int):
        field = 'concurrency' if self.sla.variable == 'concurrency' else 'request_rate'
        return self.template.model_copy(update={field: value})

    async def _evaluate(self, value: int) -> SLAEvaluation:
        load = self._load(value)
        key = load.model_dump_json()
        if key in self._cache:
            return self._cache[key][0]
        results = []
        for repetition in range(self.sla.repetitions):
            item_limit = load.request_count
            spec = ResolvedRunSpec(
                load_id=f'sla-{value}-rep-{repetition}',
                load=load,
                seed=(self.config.runtime.seed or 0) + value * 1000 + repetition,
                warmup_count=load.warmup.resolve(item_limit or 1),
                item_limit=item_limit,
            )
            results.append(await RunEngine(self.config, self.run_id, spec, self.suite_dir).run())

        passed = None
        objective = None
        if self.sla.criteria:
            repeat_passes = [self._passes(result) for result in results]
            passed = sum(repeat_passes) / len(repeat_passes) >= self.sla.pass_ratio
        else:
            objective = median(objective_value(result, self.sla.objective) for result in results)
        evaluation = SLAEvaluation(
            load_value=value,
            passed=passed,
            objective_value=objective,
            run_ids=[run.run_spec.load_id for run in results],
        )
        self._cache[key] = (evaluation, results)
        return evaluation

    def _passes(self, result: RunResult) -> bool:
        if result.summary.success_rate < 100:
            return False
        groups = []
        for group in self.sla.criteria:
            groups.append(
                all(
                    parse_criterion(expression).validate(metric_value(result, metric))
                    for metric, expression in group.items()
                )
            )
        return any(groups)

    async def _constraint_search(self) -> Optional[int]:
        lower = self.sla.lower_bound
        upper = self.sla.upper_bound
        first = await self._evaluate(lower)
        if not first.passed:
            return None
        best = lower
        cursor = lower
        failed_bound = None
        while cursor < upper:
            candidate = min(upper, cursor * 2)
            evaluation = await self._evaluate(candidate)
            if evaluation.passed:
                best = candidate
                cursor = candidate
                if candidate == upper:
                    return candidate
            else:
                failed_bound = candidate
                break
        if failed_bound is None:
            return best
        left, right = best + 1, failed_bound - 1
        while left <= right:
            middle = (left + right) // 2
            evaluation = await self._evaluate(middle)
            if evaluation.passed:
                best = middle
                left = middle + 1
            else:
                right = middle - 1
        if best < upper:
            await self._evaluate(best + 1)
        if self._is_non_monotonic():
            for value in range(lower, failed_bound + 1):
                await self._evaluate(value)
            passed = [int(item[0].load_value) for item in self._cache.values() if item[0].passed]
            return max(passed) if passed else None
        return best

    async def _objective_search(self) -> int:
        values = []
        value = self.sla.lower_bound
        while True:
            values.append(value)
            await self._evaluate(value)
            if value == self.sla.upper_bound:
                break
            value = min(self.sla.upper_bound, value * 2)
        best = self._best_objective()
        index = values.index(best)
        left = values[max(0, index - 1)]
        right = values[min(len(values) - 1, index + 1)]
        for candidate in range(left, right + 1):
            await self._evaluate(candidate)
        return self._best_objective()

    def _best_objective(self) -> int:
        evaluations = [item[0] for item in self._cache.values()]
        reverse = self.sla.objective != 'min_latency'

        def sort_key(item: SLAEvaluation):
            if reverse:
                return item.objective_value, -item.load_value
            return -item.objective_value, -item.load_value

        return int(max(evaluations, key=sort_key).load_value)

    def _is_non_monotonic(self) -> bool:
        ordered = sorted((item[0] for item in self._cache.values()), key=lambda item: item.load_value)
        seen_failure = False
        for item in ordered:
            if item.passed is False:
                seen_failure = True
            elif seen_failure and item.passed:
                return True
        return False
