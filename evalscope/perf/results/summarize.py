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
    result['minimum'] = min(data) if data else None
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
        if item.latency and item.latency > 0:
            if item.prompt_tokens is not None:
                values['input_throughput'].append(item.prompt_tokens / item.latency)
            if item.completion_tokens is not None:
                values['output_throughput'].append(item.completion_tokens / item.latency)
            if item.prompt_tokens is not None and item.completion_tokens is not None:
                values['total_throughput'].append((item.prompt_tokens + item.completion_tokens) / item.latency)
        if item.tpot and item.tpot > 0:
            values['decode_throughput'].append(1 / item.tpot)
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
    metrics: Dict[str, List[float]] = defaultdict(list)
    for items in grouped.values():
        items.sort(key=lambda item: item.turn_index if item.turn_index is not None else 0)
        starts = [item.start_time for item in items if item.start_time is not None]
        ends = [item.completed_time for item in items if item.completed_time is not None]
        if starts and ends:
            metrics['trace_latency'].append(max(ends) - min(starts))
        metrics['turns'].append(float(len(items)))
        first = items[0]
        last = items[-1]
        if first.ttft is not None:
            metrics['first_turn_ttft'].append(first.ttft)
        if first.start_time is not None and last.start_time is not None and last.ttft is not None:
            metrics['ttfat'].append(max(0.0, last.start_time + last.ttft - first.start_time))
        decode_rates = [1 / item.tpot for item in items if item.tpot and item.tpot > 0]
        if decode_rates:
            metrics['decode_throughput'].append(mean(decode_rates))
        prompt_tokens = sum(item.prompt_tokens or 0 for item in items)
        if prompt_tokens:
            metrics['cache_hit_rate'].append(sum(item.cached_tokens or 0 for item in items) / prompt_tokens * 100)
        eligible = 0
        eligible_cached = 0
        for previous, current in zip(items, items[1:]):
            previous_context = (previous.prompt_tokens or 0) + (previous.completion_tokens or 0)
            if previous_context > 0:
                eligible += previous_context
                eligible_cached += current.cached_tokens or 0
        if eligible:
            metrics['eligible_cache_hit_rate'].append(eligible_cached / eligible * 100)
    return TraceSummary(
        trace_count=len(grouped),
        averages={
            name: mean(values)
            for name, values in metrics.items()
            if values
        },
        percentiles={
            name: percentile_stats(values)
            for name, values in metrics.items()
            if values
        },
    )


def _workload_summary(
    observations: List[RequestObservation],
    last_window_seconds: float,
    steady_state_warmup_ratio: float,
) -> Optional[WorkloadSummary]:
    successful = sorted(
        (item for item in observations if item.success and item.completed_time is not None),
        key=lambda item: item.completed_time,
    )
    if not successful:
        return None
    starts = [item.start_time for item in successful if item.start_time is not None]
    if not starts:
        return None
    duration = max(item.completed_time for item in successful) - min(starts)
    if duration <= 0:
        return None

    wall_start = successful[0].start_time
    cumulative_completion = 0
    cumulative_new_prompt = 0
    cumulative_cached_prompt = 0
    points = []
    from evalscope.perf.domain.result import WorkloadTimelinePoint

    for item in successful:
        prompt = item.prompt_tokens or 0
        cached = item.cached_tokens or 0
        cumulative_completion += item.completion_tokens or 0
        cumulative_new_prompt += max(prompt - cached, 0)
        cumulative_cached_prompt += cached
        points.append(
            WorkloadTimelinePoint(
                t=max(0.0, item.completed_time - wall_start),
                cumulative_completion_tokens=cumulative_completion,
                cumulative_new_prompt_tokens=cumulative_new_prompt,
                cumulative_cached_prompt_tokens=cumulative_cached_prompt,
            )
        )
    duration = points[-1].t
    if duration <= 0:
        return None

    def rates(anchor_index: Optional[int]) -> Dict[str, float]:
        last = points[-1]
        if anchor_index is None:
            elapsed = last.t
            request_count = len(points)
            completion = last.cumulative_completion_tokens
            new_prompt = last.cumulative_new_prompt_tokens
            cached = last.cumulative_cached_prompt_tokens
        else:
            anchor = points[anchor_index]
            elapsed = last.t - anchor.t
            request_count = len(points) - anchor_index - 1
            completion = last.cumulative_completion_tokens - anchor.cumulative_completion_tokens
            new_prompt = last.cumulative_new_prompt_tokens - anchor.cumulative_new_prompt_tokens
            cached = last.cumulative_cached_prompt_tokens - anchor.cumulative_cached_prompt_tokens
        window = max(elapsed, 1e-12)
        return {
            'request_throughput': request_count / window,
            'total_prompt_token_throughput': (new_prompt + cached) / window,
            'new_prompt_token_throughput': new_prompt / window,
            'cached_prompt_token_throughput': cached / window,
            'completion_token_throughput': completion / window,
        }

    def anchor(target: float) -> int:
        chosen = 0
        for index, point in enumerate(points):
            if point.t <= target:
                chosen = index
            else:
                break
        return chosen

    last_anchor = None if duration <= last_window_seconds else anchor(duration - last_window_seconds)
    steady_anchor = anchor(duration * steady_state_warmup_ratio) if steady_state_warmup_ratio > 0 else None
    if steady_anchor == len(points) - 1:
        steady_anchor = None
    return WorkloadSummary(
        n_samples=len(successful),
        wall_time_seconds=duration,
        last_window_seconds=last_window_seconds,
        steady_state_warmup_ratio=steady_state_warmup_ratio,
        overall=rates(None),
        last_window=rates(last_anchor),
        steady_state=rates(steady_anchor),
        points=points,
    )
