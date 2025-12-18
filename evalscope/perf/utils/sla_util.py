import copy
import json
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from tabulate import tabulate
from typing import Any, Callable, Dict, List, Optional, Union

from evalscope.perf.arguments import Arguments
from evalscope.perf.utils.benchmark_util import Metrics
from evalscope.perf.utils.db_util import PercentileMetrics, average_results
from evalscope.perf.utils.rich_display import print_summary
from evalscope.utils.logger import get_logger

logger = get_logger()

MAX_CONCURRENCY = 65535  # Safety limit for concurrency in SLA auto-tuning


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


@dataclass
class SLALessThanOrEqualTo(SLACriterionBase):

    def validate(self, actual: float) -> bool:
        return actual <= self.target

    def format_cond(self, lhs: str) -> str:
        return f'{lhs} <= {self.target}'


@dataclass
class SLAGreaterThan(SLACriterionBase):

    def validate(self, actual: float) -> bool:
        return actual > self.target

    def format_cond(self, lhs: str) -> str:
        return f'{lhs} > {self.target}'


@dataclass
class SLAGreaterThanOrEqualTo(SLACriterionBase):

    def validate(self, actual: float) -> bool:
        return actual >= self.target

    def format_cond(self, lhs: str) -> str:
        return f'{lhs} >= {self.target}'


SLA_CRITERIA = {
    '<=': SLALessThanOrEqualTo,
    '>=': SLAGreaterThanOrEqualTo,
    '<': SLALessThan,
    '>': SLAGreaterThan,
}


def parse_sla_params(
    sla_params_str: Optional[Union[str, Dict[str, Any], List[Any]]]
) -> List[Dict[str, SLACriterionBase]]:
    if sla_params_str is None:
        return []

    records = []
    if isinstance(sla_params_str, dict):
        records = [sla_params_str]
    elif isinstance(sla_params_str, list):
        records = sla_params_str
    else:
        try:
            parsed = json.loads(sla_params_str)
            if isinstance(parsed, dict):
                records = [parsed]
            elif isinstance(parsed, list):
                records = parsed
            else:
                raise ValueError('SLA params must be a dictionary or a list of dictionaries')
        except (json.JSONDecodeError, TypeError):
            raise ValueError(f'Invalid JSON for --sla-params: {sla_params_str}')

    parsed_sla = []
    for record in records:
        if not isinstance(record, dict):
            continue
        criteria = {}
        for metric_key, metric_value in record.items():
            metric_value = str(metric_value).strip()
            matched = False
            for op_key in sorted(SLA_CRITERIA.keys(), key=len, reverse=True):
                if metric_value.startswith(op_key):
                    val = float(metric_value[len(op_key):])
                    criteria[metric_key] = SLA_CRITERIA[op_key](val)
                    matched = True
                    break
            if not matched:
                raise ValueError(f'Invalid operator in SLA param: {metric_value}')
        parsed_sla.append(criteria)
    return parsed_sla


def check_sla(results: Dict[str, Any], sla_criteria: List[Dict[str, SLACriterionBase]], selector: str = None) -> bool:
    metrics = results['metrics']
    percentiles_data = results['percentiles']
    prefix = f'[{selector}] ' if selector else ''

    # 1. Check Success Rate (Must be 100%)
    succeed = metrics.get(Metrics.SUCCEED_REQUESTS, 0)
    total = metrics.get(Metrics.TOTAL_REQUESTS, 0)
    if total > 0:
        success_rate = (succeed / total) * 100
    else:
        success_rate = 0.0

    if success_rate < 100.0:
        logger.warning(f'{prefix}SLA Check: Success Rate = {success_rate:.2f}% | Expect 100% | FAILED')
        return False

    # 2. Prepare values for SLA check
    values = {}
    values['avg_latency'] = metrics.get(Metrics.AVERAGE_LATENCY, 0)
    values['avg_ttft'] = metrics.get(Metrics.AVERAGE_TIME_TO_FIRST_TOKEN, 0)
    values['avg_tpot'] = metrics.get(Metrics.AVERAGE_TIME_PER_OUTPUT_TOKEN, 0)
    values['rps'] = metrics.get(Metrics.REQUEST_THROUGHPUT, 0)
    values['tps'] = metrics.get(Metrics.OUTPUT_TOKEN_THROUGHPUT, 0)

    if percentiles_data:
        try:
            p_list = percentiles_data.get(PercentileMetrics.PERCENTILES, [])
            p99_idx = p_list.index('99%')
        except ValueError:
            p99_idx = -1

        def get_p99(key):
            if p99_idx == -1:
                return 0
            lst = percentiles_data.get(key, [])
            if lst and len(lst) > p99_idx:
                val = lst[p99_idx]
                return val if isinstance(val, (int, float)) else 0
            return 0

        values['p99_latency'] = get_p99(PercentileMetrics.LATENCY)
        values['p99_ttft'] = get_p99(PercentileMetrics.TTFT)
        values['p99_tpot'] = get_p99(PercentileMetrics.TPOT)
    else:
        values['p99_latency'] = 0
        values['p99_ttft'] = 0
        values['p99_tpot'] = 0

    if not sla_criteria:
        return True

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
            logger.info(
                f"{prefix}SLA Rule {i+1} Check: {metric} = {val:.4f} | Expect {criterion.format_cond('')} | {status}"
            )

            if not passed:
                group_passed = False

        if group_passed:
            any_group_passed = True

    return any_group_passed


def run_sla_auto_tune(args: Arguments, runner: Callable[[Arguments, Optional[str]], Dict[str, Any]]):
    if not args.sla_params:
        logger.error('SLA params are required for auto-tuning.')
        return {}

    sla_params = parse_sla_params(args.sla_params)
    sla_variable = args.sla_variable

    logger.info(f'Starting SLA Auto-tune for {sla_variable}')
    logger.info(f'SLA Params: {args.sla_params}')

    # Flatten SLA criteria to check each metric independently
    target_criteria_list = []
    for group in sla_params:
        for metric, criterion in group.items():
            target_criteria_list.append({metric: criterion})

    current_val = args.parallel if sla_variable == 'parallel' else args.rate
    if isinstance(current_val, list):
        current_val = current_val[0]

    results_cache = {}
    sla_results_table = []

    def save_summary_json(cache):
        formatted = {}
        for val, res in cache.items():
            key = f'{sla_variable}_{val}'
            formatted[key] = res

        json_path = os.path.join(args.outputs_dir, 'sla_summary.json')
        try:
            with open(json_path, 'w') as f:
                json.dump(formatted, f, indent=4)
            logger.info(f'SLA summary saved to: {json_path}')
        except Exception as e:
            logger.error(f'Failed to save SLA summary json: {e}')
        return formatted

    def get_result(val):
        if val in results_cache:
            return results_cache[val]

        run_results = []
        for i in range(3):
            logger.info(f'Running {sla_variable}={val}, iteration {i+1}/3')
            run_args = copy.deepcopy(args)
            if sla_variable == 'parallel':
                run_args.parallel = val
                run_args.number = val * 2
                run_args.rate = -1
            elif sla_variable == 'rate':
                run_args.rate = val
                run_args.number = val * 2
                run_args.parallel = MAX_CONCURRENCY  # Set high concurrency for rate-limited tests
            else:
                raise ValueError(f'Unsupported SLA variable: {sla_variable}')

            subdir = f'sla_{sla_variable}_{val}_run_{i}'
            output_path = os.path.join(args.outputs_dir, 'sla_tuning', subdir)
            os.makedirs(output_path, exist_ok=True)

            res = runner(run_args, output_path)
            res_val = list(res.values())[0]
            run_results.append(res_val)

        avg_result = average_results(run_results)
        results_cache[val] = avg_result
        return avg_result

    for criteria in target_criteria_list:
        logger.info(f'Auto-tuning for criteria: {criteria}')

        def check_current_criteria(val):
            res = get_result(val)
            selector = f'{sla_variable}={val}'
            return check_sla(res, [criteria], selector)

        passed = check_current_criteria(current_val)

        lower_bound = current_val
        upper_bound = current_val

        if passed:
            logger.info('Initial run passed. Finding upper bound...')
            lower_bound = current_val
            upper_bound = current_val * 2

            while True:
                logger.info(f'Testing upper bound: {upper_bound}')
                passed = check_current_criteria(upper_bound)
                if not passed:
                    logger.info(f'Found upper bound violation at {upper_bound}')
                    break
                lower_bound = upper_bound
                upper_bound *= 2
                if upper_bound > MAX_CONCURRENCY:
                    logger.warning(f'Upper bound exceeded safety limit ({MAX_CONCURRENCY}). Stopping.')
                    break
        else:
            logger.info('Initial run failed. Finding lower bound...')
            upper_bound = current_val
            lower_bound = current_val // 2

            found_valid = False
            while lower_bound >= 1:
                logger.info(f'Testing lower bound: {lower_bound}')
                passed = check_current_criteria(lower_bound)
                if passed:
                    logger.info(f'Found valid lower bound at {lower_bound}')
                    found_valid = True
                    break

                upper_bound = lower_bound
                lower_bound //= 2

            if not found_valid:
                logger.warning(f'Even {sla_variable}=1 failed SLA for {criteria}. Cannot decrease further.')
                sla_results_table.append({
                    'Criteria': str(criteria),
                    'Variable': sla_variable,
                    'Max Satisfied': 'None',
                    'Note': 'Failed at min value'
                })
                continue

        logger.info(f'Binary search in [{lower_bound}, {upper_bound}]')
        best_val = lower_bound

        left = lower_bound + 1
        right = upper_bound - 1

        is_upper_passed = check_current_criteria(upper_bound)
        if is_upper_passed:
            best_val = upper_bound
            left = right + 1

        while left <= right:
            mid = (left + right) // 2
            logger.info(f'Binary search checking: {mid}')
            passed = check_current_criteria(mid)

            if passed:
                best_val = mid
                left = mid + 1
            else:
                right = mid - 1

        logger.info(f'SLA Auto-tune finished for {criteria}. Max {sla_variable} satisfying SLA: {best_val}')
        sla_results_table.append({
            'Criteria': str(criteria),
            'Variable': sla_variable,
            'Max Satisfied': best_val,
            'Note': 'Satisfied'
        })

    results = save_summary_json(results_cache)
    print_summary(results, args)

    if sla_results_table:
        print('\nSLA Auto-tune Summary:')
        print(tabulate(sla_results_table, headers='keys', tablefmt='grid'))

    return results
