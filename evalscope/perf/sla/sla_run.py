import copy
import json
import math
import os
import time
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union

from tabulate import tabulate

from evalscope.metrics.semantics import format_perf_value
from evalscope.perf.arguments import Arguments
from evalscope.perf.utils.db_util import average_results
from evalscope.perf.utils.perf_constants import Metrics, PercentileMetrics
from evalscope.perf.utils.perf_models import BenchmarkSummary, PercentileResult
from evalscope.perf.utils.rich_display import print_summary
from evalscope.utils.logger import get_logger

from .sla_criterion import SLACriterionBase, SLAMax, SLAMin, create_criterion, format_sla_operand
from .sla_models import SLAProbe, SLASelection

logger = get_logger()

#: SLA metric name -> perf contract field key, which is the authority for that metric's unit.
SLA_METRIC_FIELDS: Dict[str, str] = {
    'avg_latency': Metrics.AVERAGE_LATENCY,
    'avg_ttft': Metrics.AVERAGE_TIME_TO_FIRST_TOKEN,
    'avg_tpot': Metrics.AVERAGE_TIME_PER_OUTPUT_TOKEN,
    'rps': Metrics.REQUEST_THROUGHPUT,
    'tps': Metrics.OUTPUT_TOKEN_THROUGHPUT,
    'p99_latency': PercentileMetrics.LATENCY,
    'p99_ttft': PercentileMetrics.TTFT,
    'p90_ttft': PercentileMetrics.TTFT,
    'p50_ttft': PercentileMetrics.TTFT,
    'p99_tpot': PercentileMetrics.TPOT,
    'p90_tpot': PercentileMetrics.TPOT,
    'p50_tpot': PercentileMetrics.TPOT,
}

_PERCENTILE_METRICS = {
    'p99_latency': ('99%', PercentileMetrics.LATENCY, 'latency'),
    'p99_ttft': ('99%', PercentileMetrics.TTFT, 'ttft'),
    'p90_ttft': ('90%', PercentileMetrics.TTFT, 'ttft'),
    'p50_ttft': ('50%', PercentileMetrics.TTFT, 'ttft'),
    'p99_tpot': ('99%', PercentileMetrics.TPOT, 'tpot'),
    'p90_tpot': ('90%', PercentileMetrics.TPOT, 'tpot'),
    'p50_tpot': ('50%', PercentileMetrics.TPOT, 'tpot'),
}
_EMBEDDING_RERANK_METRICS = {'avg_latency', 'p99_latency', 'rps'}
_COUNT_FIELDS = (
    Metrics.TOTAL_REQUESTS,
    Metrics.SUCCEED_REQUESTS,
    Metrics.FAILED_REQUESTS,
    Metrics.STREAM_REQUESTS,
    Metrics.NON_STREAM_REQUESTS,
)


class SLAResult(dict):
    """Legacy pressure-keyed results with Python-only tuning details."""

    def __init__(self, results: Dict[str, Any], probes: List[SLAProbe], selections: List[SLASelection]) -> None:
        super().__init__(results)
        self.probes = probes
        self.selections = selections


def _format_sla_value(metric: str, value: float) -> str:
    """Render an SLA metric value in the unit its perf contract declares."""
    field_key = SLA_METRIC_FIELDS.get(metric)
    return format_perf_value(value, field_key) if field_key else f'{value:.4f}'


def parse_sla_params(
    sla_params_str: Optional[Union[str, Dict[str, Any], List[Any]]],
) -> List[Dict[str, SLACriterionBase]]:
    if sla_params_str is None:
        return []

    records = []
    if isinstance(sla_params_str, (dict, list)):
        records = sla_params_str if isinstance(sla_params_str, list) else [sla_params_str]
    else:
        try:
            parsed = json.loads(sla_params_str)
            records = parsed if isinstance(parsed, list) else [parsed]
        except (json.JSONDecodeError, TypeError):
            raise ValueError(f'Invalid JSON for --sla-params: {sla_params_str}')

    if not records:
        raise ValueError('--sla-params must contain at least one non-empty criterion group')

    parsed_sla = []
    for record in records:
        if not isinstance(record, dict) or not record:
            raise ValueError('--sla-params must contain only non-empty criterion groups')
        criteria = {}
        for name, value in record.items():
            field_key = SLA_METRIC_FIELDS.get(name)
            if field_key is None:
                raise ValueError(f"Unknown SLA metric '{name}'; supported: {', '.join(sorted(SLA_METRIC_FIELDS))}")
            criteria[name] = create_criterion(value, field_key)
        parsed_sla.append(criteria)
    extrema = [
        criterion for group in parsed_sla for criterion in group.values() if isinstance(criterion, (SLAMax, SLAMin))
    ]
    if extrema and (len(parsed_sla) != 1 or len(parsed_sla[0]) != 1):
        raise ValueError('max/min requires exactly one SLA group containing one metric')
    return parsed_sla


def get_metric_values(results: Dict[str, Any]) -> Dict[str, float]:
    """Return only metrics actually present with finite numeric values."""
    raw_metrics = results.get('metrics', {})
    raw_perc = results.get('percentiles', {})
    if isinstance(raw_metrics, BenchmarkSummary):
        raw_metrics = raw_metrics.model_dump(by_alias=True, exclude_unset=True)
    values = {}
    for name, field in SLA_METRIC_FIELDS.items():
        if name in _PERCENTILE_METRICS:
            continue
        value = raw_metrics.get(field) if isinstance(raw_metrics, dict) else None
        if _is_finite_number(value):
            values[name] = float(value)

    for name, (percentile, field, attribute) in _PERCENTILE_METRICS.items():
        value = None
        if isinstance(raw_perc, PercentileResult):
            row = next((row for row in raw_perc.rows if row.percentile == percentile), None)
            value = getattr(row, attribute, None) if row is not None else None
        elif isinstance(raw_perc, dict):
            labels = raw_perc.get(PercentileMetrics.PERCENTILES, [])
            column = raw_perc.get(field, [])
            if percentile in labels and isinstance(column, list):
                index = labels.index(percentile)
                value = column[index] if index < len(column) else None
        if _is_finite_number(value):
            values[name] = float(value)
    return values


def _is_finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _request_counts(results: Dict[str, Any]) -> Tuple[int, int]:
    raw_metrics = results.get('metrics', {})
    if isinstance(raw_metrics, BenchmarkSummary):
        raw_metrics = raw_metrics.model_dump(by_alias=True, exclude_unset=True)
    if not isinstance(raw_metrics, dict):
        return 0, 0
    total = raw_metrics.get(Metrics.TOTAL_REQUESTS)
    succeeded = raw_metrics.get(Metrics.SUCCEED_REQUESTS)
    if not all(isinstance(value, int) and not isinstance(value, bool) for value in (total, succeeded)):
        return 0, 0
    if total <= 0 or succeeded < 0 or succeeded > total:
        return 0, 0
    return total, succeeded


def check_sla(
    results: Dict[str, Any], sla_criteria: List[Dict[str, SLACriterionBase]], selector: Optional[str] = None
) -> bool:
    prefix = f'[{selector}] ' if selector else ''
    raw_metrics = results.get('metrics', {})
    if isinstance(raw_metrics, BenchmarkSummary):
        raw_metrics = raw_metrics.model_dump(by_alias=True, exclude_unset=True)
    succeed = raw_metrics.get(Metrics.SUCCEED_REQUESTS, 0) if isinstance(raw_metrics, dict) else 0
    total = raw_metrics.get(Metrics.TOTAL_REQUESTS, 0) if isinstance(raw_metrics, dict) else 0

    if not _is_finite_number(total) or not _is_finite_number(succeed) or total <= 0 or succeed != total:
        success_rate = (
            succeed / total * 100 if _is_finite_number(total) and total > 0 and _is_finite_number(succeed) else 0.0
        )
        logger.warning(f'{prefix}SLA Check: Success Rate = {success_rate:.2f}% | Expect 100% | FAILED')
        return False

    if not sla_criteria:
        return True

    return _check_criteria(get_metric_values(results), sla_criteria, prefix)


def _check_criteria(values: Dict[str, float], sla_criteria: List[Dict[str, SLACriterionBase]], prefix: str) -> bool:
    """Evaluate criterion groups using only metrics present in the observation."""
    any_group_passed = False

    for i, criteria_group in enumerate(sla_criteria):
        group_passed = True
        for metric, criterion in criteria_group.items():
            val = values.get(metric)
            if val is None:
                logger.warning(f'{prefix}Metric {metric} not found in results.')
                group_passed = False
                continue

            passed = criterion.validate(val)
            status = 'PASSED' if passed else 'FAILED'
            actual = format_sla_operand(val, SLA_METRIC_FIELDS[metric])
            logger.info(
                f'{prefix}SLA Rule {i + 1} Check: {metric} = {actual} | Expect {criterion.format_cond("")} | {status}'
            )
            if not passed:
                group_passed = False

        if group_passed:
            any_group_passed = True

    return any_group_passed


class SLAAutoTuner:
    def __init__(self, args: Arguments, runner: Callable[[Arguments, Optional[str]], Dict[str, Any]]) -> None:
        self.args = args
        self.runner = runner
        self.sla_variable = args.sla_variable
        self.results_cache: Dict[int, SLAProbe] = {}
        self.selections: List[SLASelection] = []
        self.metric_names: set[str] = set()
        self.criteria_groups: List[Dict[str, SLACriterionBase]] = []
        self.upper_bound = args.sla_upper_bound
        self.lower_bound = args.sla_lower_bound
        self.fixed_parallel = args.sla_fixed_parallel if args.sla_fixed_parallel is not None else args.sla_upper_bound

    def tune(self) -> SLAResult:
        sla_params = parse_sla_params(self.args.sla_params)
        if not sla_params:
            raise ValueError('--sla-params is required for SLA auto-tuning')
        self.criteria_groups = sla_params
        self.metric_names = {name for group in sla_params for name in group}
        if Metrics.is_embedding_or_rerank(self.args.api):
            unsupported = self.metric_names - _EMBEDDING_RERANK_METRICS
            if unsupported:
                raise ValueError(f'Metrics unavailable for {self.args.api}: {", ".join(sorted(unsupported))}')
        logger.info(f'Starting SLA Auto-tune for {self.sla_variable}')
        logger.info(f'SLA Range: [{self.lower_bound}, {self.upper_bound}]')
        logger.info(f'SLA Params: {self.args.sla_params}')

        # sla_params is a list of criterion groups. Each group is evaluated independently:
        # - Within one group (dict): ALL metrics must pass → AND logic.
        # - Each group in the list runs its own binary search and produces its own result row.
        #
        # Examples:
        #   AND: [{"avg_ttft": "<=2s", "avg_tpot": "<=50ms"}]         → single group, both required
        #   OR:  [{"avg_ttft": "<=2s"}, {"avg_tpot": "<=50ms"}]       → two independent searches
        #
        # Special case: a single-group single-metric max/min triggers optimization mode.
        current_val = self.args.parallel if self.sla_variable == 'parallel' else self.args.rate
        if isinstance(current_val, list):
            current_val = current_val[0]

        # Ensure current_val is an int within bounds so binary search and cache keys are consistent
        current_val = int(max(self.lower_bound, min(current_val, self.upper_bound)))

        # Check if this is a single-group single-metric optimization (max/min)
        if len(sla_params) == 1 and len(sla_params[0]) == 1:
            opt_metric, opt_mode = self._get_optimization_mode(sla_params[0])
            if opt_mode:
                self._tune_optimization(current_val, opt_metric, opt_mode)
                return self._finalize_results()

        # General case: each group runs an independent binary search and produces its own result
        for group in sla_params:
            criteria_desc = ' AND '.join(f'{k} {v}' for k, v in group.items())
            logger.info(f'Auto-tuning for criteria group: {criteria_desc}')
            self._tune_constraint(current_val, group)

        return self._finalize_results()

    def _finalize_results(self) -> SLAResult:
        for probe in self.results_cache.values():
            probe.group_passes = {
                str(index): self._group_passes(probe, group)
                for index, group in enumerate(self.criteria_groups, start=1)
            }
        results = SLAResult(
            {f'{self.sla_variable}_{value}': probe.averaged_result for value, probe in self.results_cache.items()},
            probes=list(self.results_cache.values()),
            selections=self.selections,
        )
        self._save_summary(results)
        print_summary(results, self.args)
        logger.info('SLA Auto-tune Summary:\n' + format_sla_tables(results, tablefmt='simple_grid'))
        return results

    def _get_optimization_mode(self, criteria: Dict[str, SLACriterionBase]) -> Tuple[Optional[str], Optional[str]]:
        for m, c in criteria.items():
            if isinstance(c, SLAMax):
                return m, 'max'
            if isinstance(c, SLAMin):
                return m, 'min'
        return None, None

    def _compute_number(self, val: int) -> int:
        """Compute the number of requests based on val and sla_number_multiplier.

        If sla_number_multiplier is set, number = round(val * sla_number_multiplier).
        Defaults to val * 2 if not set.
        """
        multiplier = self.args.sla_number_multiplier
        if multiplier is None:
            return max(1, round(val * 2))
        return max(1, round(val * multiplier))

    def _get_probe(self, val: int) -> SLAProbe:
        if val in self.results_cache:
            return self.results_cache[val]

        run_results = []
        total_requests = 0
        succeeded_requests = 0
        count_sums = {field: 0 for field in _COUNT_FIELDS}
        reasons = []
        run_values = []
        for i in range(self.args.sla_num_runs):
            logger.info(f'Running {self.sla_variable}={val}, iteration {i + 1}/{self.args.sla_num_runs}...')
            run_args = copy.deepcopy(self.args)

            if self.sla_variable == 'parallel':
                run_args.parallel = val
                run_args.number = self._compute_number(val)
                run_args.rate = -1
            elif self.sla_variable == 'rate':
                run_args.rate = val
                run_args.number = self._compute_number(val)
                run_args.parallel = self.fixed_parallel
            else:
                raise ValueError(f'Unsupported SLA variable: {self.sla_variable}')

            subdir = f'sla_{self.sla_variable}_{val}_run_{i}'
            output_path = os.path.join(self.args.outputs_dir, 'sla_tuning', subdir)
            os.makedirs(output_path, exist_ok=True)

            res = self.runner(run_args, output_path)
            if len(res) != 1:
                raise ValueError(f'SLA runner must return exactly one result at {self.sla_variable}={val}')
            run_result = next(iter(res.values()))
            run_results.append(run_result)
            run_metrics = run_result.get('metrics', {})
            if isinstance(run_metrics, BenchmarkSummary):
                run_metrics = run_metrics.model_dump(by_alias=True, exclude_unset=True)
            if isinstance(run_metrics, dict):
                for field in _COUNT_FIELDS:
                    value = run_metrics.get(field)
                    if isinstance(value, int) and not isinstance(value, bool):
                        count_sums[field] += value
            total, succeeded = _request_counts(run_result)
            total_requests += total
            succeeded_requests += succeeded
            if not total or succeeded != total:
                reasons.append(f'run {i + 1}: {succeeded}/{total} requests succeeded')
            values = get_metric_values(run_result)
            run_values.append(values)
            for metric in sorted(self.metric_names - values.keys()):
                reasons.append(f'run {i + 1}: missing or non-finite {metric}')

            # Advance dataset_offset on the source args so the next deepcopy
            # picks up the new offset, preventing KV-cache hits across runs.
            self.args.dataset_offset += run_args.number

            if i < self.args.sla_num_runs - 1:
                logger.info(f'Sleeping {self.args.sleep_interval} seconds before next run...')
                time.sleep(self.args.sleep_interval)

        avg_result = average_results(run_results)
        averaged_metrics = avg_result.get('metrics', {})
        averaged_metrics.update(count_sums)
        averaged_metrics[Metrics.TOTAL_REQUESTS] = total_requests
        averaged_metrics[Metrics.SUCCEED_REQUESTS] = succeeded_requests
        averaged_metrics[Metrics.FAILED_REQUESTS] = total_requests - succeeded_requests
        metric_values = {
            metric: sum(values[metric] for values in run_values) / len(run_values)
            for metric in self.metric_names
            if all(metric in values for values in run_values)
        }
        probe = SLAProbe(
            value=val,
            averaged_result=avg_result,
            total_requests=total_requests,
            succeeded_requests=succeeded_requests,
            success_rate=succeeded_requests / total_requests * 100 if total_requests else 0.0,
            valid=not reasons,
            reasons=reasons,
            metric_values=metric_values,
        )
        self.results_cache[val] = probe
        return probe

    def _optimization_value(self, val: int, opt_metric: str) -> Optional[float]:
        """Exclude failed runs and missing metrics from the optimization objective."""
        probe = self._get_probe(val)
        if probe.total_requests == 0 or probe.succeeded_requests != probe.total_requests:
            return None
        return probe.metric_values.get(opt_metric)

    @staticmethod
    def _group_passes(probe: SLAProbe, criteria: Dict[str, SLACriterionBase]) -> bool:
        if probe.total_requests == 0 or probe.succeeded_requests != probe.total_requests:
            return False
        return all(
            metric in probe.metric_values and criterion.validate(probe.metric_values[metric])
            for metric, criterion in criteria.items()
        )

    def _check_probe(self, val: int, criteria: List[Dict[str, SLACriterionBase]]) -> bool:
        probe = self._get_probe(val)
        selector = f'{self.sla_variable}={val}'
        if probe.total_requests == 0 or probe.succeeded_requests != probe.total_requests:
            logger.warning(f'[{selector}] SLA Check: {"; ".join(probe.reasons)} | FAILED')
            return False
        return _check_criteria(probe.metric_values, criteria, f'[{selector}] ')

    @staticmethod
    def _improves(val: Optional[float], best: Optional[float], opt_mode: str) -> bool:
        if val is None:
            return False
        if best is None:
            return True
        return val > best if opt_mode == 'max' else val < best

    def _tune_optimization(self, start_val: int, opt_metric: str, opt_mode: str) -> None:
        logger.info(f'Optimization mode: {opt_mode} for {opt_metric}')

        def score(value: int) -> Optional[float]:
            return self._optimization_value(value, opt_metric)

        def not_worse(candidate: Optional[float], baseline: Optional[float]) -> bool:
            if candidate is None:
                return False
            if baseline is None:
                return True
            return candidate >= baseline if opt_mode == 'max' else candidate <= baseline

        start_score = score(start_val)
        if start_score is None:
            # Under the documented valid-prefix assumption, a failed start can
            # still have a useful lower-pressure interval.
            if score(self.lower_bound) is not None:
                left, right = self.lower_bound, start_val - 1
                while left < right:
                    mid = (left + right + 1) // 2
                    if score(mid) is not None:
                        left = mid
                    else:
                        right = mid - 1
                bracket = (self.lower_bound, left)
            else:
                bracket = None
        else:
            left_score = score(start_val - 1) if start_val > self.lower_bound else None
            right_score = score(start_val + 1) if start_val < self.upper_bound else None
            if self._improves(left_score, start_score, opt_mode):
                direction = -1
            elif self._improves(right_score, start_score, opt_mode):
                direction = 1
            elif left_score == start_score and left_score is not None:
                direction = -1
            elif right_score == start_score and right_score is not None:
                direction = 1
            else:
                direction = 0

            if direction:
                previous = start_val
                current = start_val + direction
                step = 1
                bracket = tuple(sorted((previous, current)))
                while self.lower_bound <= current + direction * step <= self.upper_bound:
                    candidate = current + direction * step
                    candidate_score = score(candidate)
                    if not_worse(candidate_score, score(current)):
                        previous = current
                        current = candidate
                        bracket = tuple(sorted((previous, current)))
                        step *= 2
                    else:
                        bracket = tuple(sorted((previous, candidate)))
                        break
                else:
                    edge = self.upper_bound if direction > 0 else self.lower_bound
                    if current != edge:
                        score(edge)
                        bracket = tuple(sorted((previous, edge)))
            else:
                bracket = (start_val, start_val)

        # Adjacent values reveal the local slope; comparison with the historical
        # best cannot safely discard half of a single-peaked interval.
        if bracket is not None:
            left, right = bracket
            while left < right:
                mid = (left + right) // 2
                current_score, next_score = score(mid), score(mid + 1)
                if current_score is None:
                    if next_score is None:
                        right = mid
                    else:
                        left = mid + 1
                elif next_score is None or not_worse(current_score, next_score):
                    right = mid
                else:
                    left = mid + 1

        candidates = [
            (value, probe.metric_values[opt_metric])
            for value, probe in self.results_cache.items()
            if self._group_passes(probe, self.criteria_groups[0])
        ]
        if candidates:
            best_sla_val, best_metric_val = min(
                candidates,
                key=lambda item: (-item[1], item[0]) if opt_mode == 'max' else (item[1], item[0]),
            )
            note = f'Best observed {opt_metric}: {_format_sla_value(opt_metric, best_metric_val)}'
        else:
            best_sla_val, best_metric_val = None, None
            note = 'No run passed the 100% success gate with the required metric'
        self._record_selection(
            criteria={opt_metric: opt_mode},
            mode=opt_mode,
            selected_value=best_sla_val,
            observed_metric=best_metric_val,
            note=note,
            assumption='single-peaked objective within a valid low-pressure prefix',
        )

    def _record_selection(
        self,
        criteria: Dict[str, str],
        mode: Literal['constraint', 'max', 'min'],
        selected_value: Optional[int],
        observed_metric: Optional[float],
        note: str,
        assumption: str,
    ) -> None:
        self.selections.append(
            SLASelection(
                criteria=criteria,
                mode=mode,
                selected_value=selected_value,
                observed_metric=observed_metric,
                status='best_observed' if selected_value is not None else 'none',
                reason=note,
                assumption=assumption,
            )
        )

    def _tune_constraint(
        self,
        start_val: int,
        criteria: Dict[str, SLACriterionBase],
    ) -> None:
        """Tune one criterion group under the monotonic-satisfaction assumption."""
        sla_criteria = [criteria]
        criteria_desc = ', '.join(f'{key} {value}' for key, value in criteria.items())

        best_observed: Optional[int] = None

        def check(val: int) -> bool:
            nonlocal best_observed
            passed = self._check_probe(val, sla_criteria)
            if passed:
                best_observed = val if best_observed is None else max(best_observed, val)
            return passed

        passed = check(start_val)
        lower, upper = start_val, start_val

        if passed:
            logger.info('Initial run passed. Finding upper bound...')
            upper = min(start_val * 2, self.upper_bound)
            while upper <= self.upper_bound:
                logger.info(f'Testing upper bound: {upper}')
                if not check(upper):
                    logger.info(f'Found upper bound violation at {upper}')
                    break
                lower = upper
                if upper == self.upper_bound:
                    logger.info(f'Reached upper bound limit: {self.upper_bound}')
                    break
                upper = min(upper * 2, self.upper_bound)
        else:
            logger.info('Initial run failed. Finding lower bound...')
            upper = start_val
            lower = max(start_val // 2, self.lower_bound)
            found_valid = False
            while lower >= self.lower_bound:
                logger.info(f'Testing lower bound: {lower}')
                if check(lower):
                    logger.info(f'Found valid lower bound at {lower}')
                    found_valid = True
                    break
                upper = lower
                if lower == self.lower_bound:
                    break
                lower = max(lower // 2, self.lower_bound)

            if not found_valid:
                logger.warning(f'Even {self.sla_variable}={self.lower_bound} failed SLA for {criteria_desc}.')
                self._record_selection(
                    criteria={key: str(value) for key, value in criteria.items()},
                    mode='constraint',
                    selected_value=None,
                    observed_metric=None,
                    note=f'Failed at lower bound ({self.lower_bound})',
                    assumption='SLA satisfaction is monotonic as pressure increases',
                )
                return

        # Binary search
        logger.info(f'Binary search in [{lower}, {upper}]')
        best_val = lower
        left, right = lower + 1, upper - 1

        # Check upper bound if it was passed (edge case where loop broke due to max concurrency)
        if check(upper):
            best_val = upper
            left = right + 1

        while left <= right:
            mid = (left + right) // 2
            logger.info(f'Binary search checking: {mid}')
            if check(mid):
                best_val = mid
                left = mid + 1
            else:
                right = mid - 1

        best_val = max(best_val, best_observed) if best_observed is not None else best_val
        logger.info(f'SLA Auto-tune finished. Criteria: {criteria_desc}. Best observed {self.sla_variable}: {best_val}')
        self._record_selection(
            criteria={key: str(value) for key, value in criteria.items()},
            mode='constraint',
            selected_value=best_val,
            observed_metric=None,
            note='Satisfied at the selected tested pressure',
            assumption='SLA satisfaction is monotonic as pressure increases',
        )

    def _save_summary(self, result: SLAResult) -> None:
        json_path = os.path.join(self.args.outputs_dir, 'sla_summary.json')
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=4)
        logger.info(f'SLA summary saved to: {json_path}')


def format_sla_tables(result: SLAResult, tablefmt: str = 'pipe') -> str:
    """Render decisions from the same exact counts stored in the SLA result."""
    probe_rows = [
        [
            probe.value,
            f'{probe.succeeded_requests}/{probe.total_requests}',
            'VALID' if probe.valid else 'FAILED',
            ', '.join(f'{group}:{"PASS" if passed else "FAIL"}' for group, passed in probe.group_passes.items()) or '-',
            ', '.join(probe.reasons) if probe.reasons else '-',
        ]
        for probe in result.probes
    ]
    selection_rows = [
        [
            ', '.join(f'{name} {target}' for name, target in selection.criteria.items()),
            selection.selected_value if selection.selected_value is not None else 'None',
            selection.status,
            selection.reason,
        ]
        for selection in result.selections
    ]
    probes = tabulate(
        probe_rows, headers=['Pressure', 'Succeeded/Total', 'Run status', 'Groups', 'Reason'], tablefmt=tablefmt
    )
    selections = tabulate(selection_rows, headers=['Criteria', 'Selected', 'Status', 'Reason'], tablefmt=tablefmt)
    return f'Probes:\n{probes}\n\nSelections:\n{selections}'


def run_sla_auto_tune(args: Arguments, runner: Callable[[Arguments, Optional[str]], Dict[str, Any]]) -> SLAResult:
    tuner = SLAAutoTuner(args, runner)
    return tuner.tune()
