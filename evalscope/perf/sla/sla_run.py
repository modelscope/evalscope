import copy
import json
import os
import time
from tabulate import tabulate
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from evalscope.perf.arguments import Arguments
from evalscope.perf.utils.benchmark_util import Metrics
from evalscope.perf.utils.db_util import PercentileMetrics, average_results
from evalscope.perf.utils.rich_display import print_summary
from evalscope.utils.logger import get_logger
from .sla_criterion import SLACriterionBase, SLAMax, SLAMin, create_criterion

logger = get_logger()


def parse_sla_params(
    sla_params_str: Optional[Union[str, Dict[str, Any], List[Any]]]
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

    parsed_sla = []
    for record in records:
        if not isinstance(record, dict):
            continue
        criteria = {k: create_criterion(v) for k, v in record.items()}
        parsed_sla.append(criteria)
    return parsed_sla


def get_metric_values(results: Dict[str, Any]) -> Dict[str, float]:
    metrics = results.get('metrics', {})
    percentiles_data = results.get('percentiles', {})

    values = {
        'avg_latency': metrics.get(Metrics.AVERAGE_LATENCY, 0),
        'avg_ttft': metrics.get(Metrics.AVERAGE_TIME_TO_FIRST_TOKEN, 0),
        'avg_tpot': metrics.get(Metrics.AVERAGE_TIME_PER_OUTPUT_TOKEN, 0),
        'rps': metrics.get(Metrics.REQUEST_THROUGHPUT, 0),
        'tps': metrics.get(Metrics.OUTPUT_TOKEN_THROUGHPUT, 0),
    }

    p99_idx = -1
    if percentiles_data:
        try:
            p99_idx = percentiles_data.get(PercentileMetrics.PERCENTILES, []).index('99%')
        except ValueError:
            pass

    def get_p99(key):
        if p99_idx == -1:
            return 0
        lst = percentiles_data.get(key, [])
        return lst[p99_idx] if lst and len(lst) > p99_idx and isinstance(lst[p99_idx], (int, float)) else 0

    values['p99_latency'] = get_p99(PercentileMetrics.LATENCY)
    values['p99_ttft'] = get_p99(PercentileMetrics.TTFT)
    values['p99_tpot'] = get_p99(PercentileMetrics.TPOT)

    return values


def check_sla(results: Dict[str, Any], sla_criteria: List[Dict[str, SLACriterionBase]], selector: str = None) -> bool:
    metrics = results.get('metrics', {})
    prefix = f'[{selector}] ' if selector else ''

    # 1. Check Success Rate (Must be 100%)
    succeed = metrics.get(Metrics.SUCCEED_REQUESTS, 0)
    total = metrics.get(Metrics.TOTAL_REQUESTS, 0)
    success_rate = (succeed / total * 100) if total > 0 else 0.0

    if success_rate < 100.0:
        logger.warning(f'{prefix}SLA Check: Success Rate = {success_rate:.2f}% | Expect 100% | FAILED')
        return False

    if not sla_criteria:
        return True

    # 2. Prepare values for SLA check
    values = get_metric_values(results)
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


class SLAAutoTuner:

    def __init__(self, args: Arguments, runner: Callable[[Arguments, Optional[str]], Dict[str, Any]]):
        self.args = args
        self.runner = runner
        self.sla_variable = args.sla_variable
        self.results_cache = {}
        self.sla_results_table = []
        self.upper_bound = args.sla_upper_bound
        self.lower_bound = args.sla_lower_bound

    def tune(self) -> Dict[str, Any]:
        if not self.args.sla_params:
            logger.error('SLA params are required for auto-tuning.')
            return {}

        sla_params = parse_sla_params(self.args.sla_params)
        logger.info(f'Starting SLA Auto-tune for {self.sla_variable}')
        logger.info(f'SLA Range: [{self.lower_bound}, {self.upper_bound}]')
        logger.info(f'SLA Params: {self.args.sla_params}')

        # Flatten SLA criteria to check each metric independently
        target_criteria_list = [{k: v} for group in sla_params for k, v in group.items()]

        current_val = self.args.parallel if self.sla_variable == 'parallel' else self.args.rate
        if isinstance(current_val, list):
            current_val = current_val[0]

        # Ensure current_val is within bounds
        current_val = max(self.lower_bound, min(current_val, self.upper_bound))

        for criteria in target_criteria_list:
            logger.info(f'Auto-tuning for criteria: {criteria}')

            # Identify optimization mode
            opt_metric, opt_mode = self._get_optimization_mode(criteria)

            if opt_mode:
                self._tune_optimization(current_val, opt_metric, opt_mode)
            else:
                self._tune_constraint(current_val, criteria)

        results = self._save_summary()
        print_summary(results, self.args)

        if self.sla_results_table:
            logger.info('SLA Auto-tune Summary:\n' + tabulate(self.sla_results_table, headers='keys', tablefmt='grid'))

        return results

    def _get_optimization_mode(self, criteria: Dict[str, SLACriterionBase]) -> Tuple[Optional[str], Optional[str]]:
        for m, c in criteria.items():
            if isinstance(c, SLAMax):
                return m, 'max'
            if isinstance(c, SLAMin):
                return m, 'min'
        return None, None

    def _get_result(self, val: int) -> Dict[str, Any]:
        if val in self.results_cache:
            return self.results_cache[val]

        run_results = []
        for i in range(self.args.sla_num_runs):
            logger.info(f'Running {self.sla_variable}={val}, iteration {i+1}/{self.args.sla_num_runs}...')
            run_args = copy.deepcopy(self.args)

            if self.sla_variable == 'parallel':
                run_args.parallel = val
                run_args.number = val * 2
                run_args.rate = -1
            elif self.sla_variable == 'rate':
                run_args.rate = val
                run_args.number = val * 2
                run_args.parallel = self.upper_bound
            else:
                raise ValueError(f'Unsupported SLA variable: {self.sla_variable}')

            subdir = f'sla_{self.sla_variable}_{val}_run_{i}'
            output_path = os.path.join(self.args.outputs_dir, 'sla_tuning', subdir)
            os.makedirs(output_path, exist_ok=True)

            res = self.runner(run_args, output_path)
            run_results.append(list(res.values())[0])

            if i < self.args.sla_num_runs - 1:
                logger.info(f'Sleeping {self.args.sleep_interval} seconds before next run...')
                time.sleep(self.args.sleep_interval)

        avg_result = average_results(run_results)
        self.results_cache[val] = avg_result
        return avg_result

    def _tune_optimization(self, start_val: int, opt_metric: str, opt_mode: str):
        logger.info(f'Optimization mode: {opt_mode} for {opt_metric}')

        best_sla_val = start_val
        curr = start_val

        # Initial run
        res = self._get_result(curr)
        best_metric_val = get_metric_values(res).get(opt_metric, 0)

        lower_bound = curr
        upper_bound = None

        # Phase 1: Find bounds by doubling
        while True:
            next_sla = curr * 2
            if next_sla > self.upper_bound:
                if curr < self.upper_bound:
                    next_sla = self.upper_bound
                else:
                    logger.info(f'Reached max concurrency limit: {self.upper_bound}')
                    break

            if next_sla == curr:
                break

            res = self._get_result(next_sla)
            next_val = get_metric_values(res).get(opt_metric, 0)
            logger.info(f'Optimization step: {self.sla_variable}={next_sla}, {opt_metric}={next_val}')

            improved = (next_val > best_metric_val) if opt_mode == 'max' else (next_val < best_metric_val)

            if improved:
                best_metric_val = next_val
                best_sla_val = next_sla
                curr = next_sla
                lower_bound = curr
                if curr == self.upper_bound:
                    break
            else:
                upper_bound = next_sla
                logger.info(f'Metric {opt_metric} did not improve. Peak is between {lower_bound} and {upper_bound}.')
                break

        # Phase 2: Binary search for peak
        if upper_bound is not None and upper_bound > lower_bound:
            logger.info(f'Binary search for optimization in [{lower_bound}, {upper_bound}]')
            left, right = lower_bound, upper_bound

            while left < right - 1:
                mid = (left + right) // 2
                res = self._get_result(mid)
                val = get_metric_values(res).get(opt_metric, 0)
                logger.info(f'Binary search checking: {self.sla_variable}={mid}, {opt_metric}={val}')

                improved = (val > best_metric_val) if opt_mode == 'max' else (val < best_metric_val)

                if improved:
                    best_metric_val = val
                    best_sla_val = mid
                    left = mid
                else:
                    right = mid

        self.sla_results_table.append({
            'Criteria': f'{opt_metric} -> {opt_mode}',
            'Variable': self.sla_variable,
            'Max Satisfied': best_sla_val,
            'Note': f'Best {opt_metric}: {best_metric_val:.4f}'
        })

    def _tune_constraint(self, start_val: int, criteria: Dict[str, SLACriterionBase]):

        def check(val):
            return check_sla(self._get_result(val), [criteria], f'{self.sla_variable}={val}')

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
                logger.warning(f'Even {self.sla_variable}={self.lower_bound} failed SLA for {criteria}.')
                self.sla_results_table.append({
                    'Criteria': ', '.join([f'{k} {v}' for k, v in criteria.items()]),
                    'Variable': self.sla_variable,
                    'Max Satisfied': 'None',
                    'Note': f'Failed at lower bound ({self.lower_bound})'
                })
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

        logger.info(f'SLA Auto-tune finished for {criteria}. Max {self.sla_variable}: {best_val}')
        self.sla_results_table.append({
            'Criteria': ', '.join([f'{k} {v}' for k, v in criteria.items()]),
            'Variable': self.sla_variable,
            'Max Satisfied': best_val,
            'Note': 'Satisfied'
        })

    def _save_summary(self) -> Dict[str, Any]:
        formatted = {f'{self.sla_variable}_{val}': res for val, res in self.results_cache.items()}
        json_path = os.path.join(self.args.outputs_dir, 'sla_summary.json')
        try:
            with open(json_path, 'w') as f:
                json.dump(formatted, f, indent=4)
            logger.info(f'SLA summary saved to: {json_path}')
        except Exception as e:
            logger.error(f'Failed to save SLA summary json: {e}')
        return formatted


def run_sla_auto_tune(args: Arguments, runner: Callable[[Arguments, Optional[str]], Dict[str, Any]]):
    tuner = SLAAutoTuner(args, runner)
    return tuner.tune()
