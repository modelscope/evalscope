from __future__ import annotations

import math
from collections import defaultdict
from statistics import mean
from typing import Callable, Dict, Iterable, List, Optional

from evalscope.perf.domain.observation import RequestObservation
from evalscope.perf.domain.result import PercentileStats, RunSummary, TraceSummary, WorkloadSummary
from evalscope.perf.results.store import ResultStore

_PERCENTILES = (1, 5, 10, 25, 50, 75, 90, 95, 99)


def _percentile(values: List[float], percentile: int) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(values)
    rank = (len(ordered) - 1) * percentile / 100
    low = math.floor(rank)
    high = math.ceil(rank)
    if low == high:
        return ordered[low]
    weight = rank - low
    return ordered[low] * (1 - weight) + ordered[high] * weight


def percentile_stats(values: Iterable[float]) -> PercentileStats:
    data = [float(value) for value in values if value is not None and math.isfinite(value)]
    result = {f'p{p}': _percentile(data, p) for p in _PERCENTILES}
    result['maximum'] = max(data) if data else None
    return PercentileStats(**result)


def _metric_values(observations: Iterable[RequestObservation]) -> Dict[str, List[float]]:
    values: Dict[str, List[float]] = defaultdict(list)
    for item in observations:
        if not item.success:
            continue
        if item.latency is not None:
            values['latency'].append(item.latency)
        if item.ttft is not None:
            values['ttft'].append(item.ttft * 1000)
        if item.tpot is not None:
            values['tpot'].append(item.tpot * 1000)
        values['itl'].extend(value * 1000 for value in item.inter_token_latencies)
        if item.arrival_lag is not None:
            values['arrival_lag'].append(item.arrival_lag * 1000)
        if item.prompt_tokens is not None:
            values['input_tokens'].append(float(item.prompt_tokens))
        if item.completion_tokens is not None:
            values['output_tokens'].append(float(item.completion_tokens))
        if item.cached_tokens is not None and item.prompt_tokens:
            values['cache_hit_rate'].append(min(1.0, item.cached_tokens / item.prompt_tokens))
        if item.ttft is not None and item.turn_index is not None:
            name = 'first_turn_ttft' if item.is_first_turn else 'subsequent_turn_ttft'
            values[name].append(item.ttft * 1000)
        if item.completion_tokens and item.completion_tokens > 1 and len(item.chunk_times) > 1:
            decoded = (item.completion_tokens - 1) / (len(item.chunk_times) - 1)
            if decoded > 1:
                values['decoded_tokens_per_iter'].append(decoded)
                values['approx_spec_acceptance_rate'].append(1 - 1 / decoded)
        if item.accepted_draft_tokens is not None and item.proposed_draft_tokens:
            values['draft_acceptance_rate'].append(item.accepted_draft_tokens / item.proposed_draft_tokens)
    return values


def summarize_store(
    store: ResultStore,
    last_window_seconds: float = 10.0,
    steady_state_warmup_ratio: float = 0.1,
) -> tuple[RunSummary, Dict[str, PercentileStats], Optional[TraceSummary], Optional[WorkloadSummary]]:
    observations = [item for item in store.observations() if not item.is_warmup]
    total = len(observations)
    succeeded = sum(item.success for item in observations)
    dropped = sum(item.dropped for item in observations)
    failed = total - succeeded - dropped
    started = [item.start_time for item in observations if item.start_time is not None]
    completed = [item.completed_time for item in observations if item.completed_time is not None]
    duration = max(completed) - min(started) if started and completed else 0.0
    prompt_tokens = sum(item.prompt_tokens or 0 for item in observations if item.success)
    completion_tokens = sum(item.completion_tokens or 0 for item in observations if item.success)
    metric_values = _metric_values(observations)
    averages = {name: mean(values) for name, values in metric_values.items() if values}
    cached_tokens = sum(item.cached_tokens or 0 for item in observations if item.success)
    if prompt_tokens:
        averages['cache_hit_rate'] = min(1.0, cached_tokens / prompt_tokens)
    summary = RunSummary(
        total=total,
        succeeded=succeeded,
        failed=max(0, failed),
        dropped=dropped,
        success_rate=(succeeded / total * 100) if total else 0,
        duration_seconds=max(0.0, duration),
        request_throughput=(succeeded / duration) if duration > 0 else 0,
        input_token_throughput=(prompt_tokens / duration) if duration > 0 else None,
        output_token_throughput=(completion_tokens / duration) if duration > 0 else None,
        total_token_throughput=((prompt_tokens + completion_tokens) / duration) if duration > 0 else None,
        average_input_tokens=(prompt_tokens / succeeded) if succeeded else None,
        average_output_tokens=(completion_tokens / succeeded) if succeeded else None,
        averages=averages,
    )
    percentiles = {name: percentile_stats(values) for name, values in metric_values.items() if values}
    trace_summary = _trace_summary(observations)
    workload_summary = _workload_summary(observations, last_window_seconds, steady_state_warmup_ratio)
    return summary, percentiles, trace_summary, workload_summary


def _trace_summary(observations: List[RequestObservation]) -> Optional[TraceSummary]:
    grouped: Dict[str, List[RequestObservation]] = defaultdict(list)
    for item in observations:
        if item.trace_id:
            grouped[item.trace_id].append(item)
    if not grouped:
        return None
    trace_latencies = []
    trace_turns = []
    for items in grouped.values():
        starts = [item.start_time for item in items if item.start_time is not None]
        ends = [item.completed_time for item in items if item.completed_time is not None]
        if starts and ends:
            trace_latencies.append(max(ends) - min(starts))
        trace_turns.append(float(len(items)))
    return TraceSummary(
        trace_count=len(grouped),
        averages={
            'trace_latency': mean(trace_latencies) if trace_latencies else 0,
            'turns': mean(trace_turns),
        },
        percentiles={
            'trace_latency': percentile_stats(trace_latencies),
            'turns': percentile_stats(trace_turns),
        },
    )


def _workload_summary(
    observations: List[RequestObservation],
    last_window_seconds: float,
    steady_state_warmup_ratio: float,
) -> Optional[WorkloadSummary]:
    successful = [item for item in observations if item.success and item.completed_time is not None]
    if not successful:
        return None
    starts = [item.start_time for item in successful if item.start_time is not None]
    if not starts:
        return None
    duration = max(item.completed_time for item in successful) - min(starts)
    if duration <= 0:
        return None

    def rates(items: List[RequestObservation], window_start: float, window_end: float) -> Dict[str, float]:
        window = max(window_end - window_start, 1e-12)
        output = sum(item.completion_tokens or 0 for item in items)
        prompt = sum(item.prompt_tokens or 0 for item in items)
        return {
            'request_throughput': len(items) / window,
            'input_token_throughput': prompt / window,
            'output_token_throughput': output / window,
            'total_token_throughput': (prompt + output) / window,
        }

    start = min(starts)
    end = max(item.completed_time for item in successful)
    last_start = max(start, end - last_window_seconds)
    steady_start = start + duration * steady_state_warmup_ratio
    last_items = [item for item in successful if item.completed_time >= last_start]
    steady_items = [item for item in successful if item.completed_time >= steady_start]
    return WorkloadSummary(
        overall=rates(successful, start, end),
        last_window=rates(last_items, last_start, end),
        steady_state=rates(steady_items, steady_start, end),
    )
