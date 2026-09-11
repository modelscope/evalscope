# SPDX-License-Identifier: Apache-2.0
"""Normalized AgentX result model with provenance and validity."""

import os
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from evalscope.perf.agentx.config import AGENTX_DATASET_REPOS
from evalscope.perf.agentx.runner import AGGREGATE_EXPORT, PER_RUN_EXPORT, parse_profile_export


class AgentxResult(BaseModel):
    """Normalized view of one AgentX run.

    Metric fields stay ``None`` when AIPerf did not report them — missing
    metrics must remain unavailable rather than be reported as zero.
    """

    # Provenance
    aiperf_version: Optional[str] = None
    aiperf_command: Optional[List[str]] = None
    dataset_variant: Optional[str] = None
    dataset_repo: Optional[str] = None
    random_seed: Optional[int] = None
    model: Optional[str] = None
    num_gpus: Optional[int] = None

    # Validity
    submission_valid: Optional[bool] = None
    """AIPerf's own scenario-validity stamp; ``None`` when absent (no --scenario)."""
    submission_invalid_reasons: List[str] = Field(default_factory=list)
    canonical: Optional[bool] = None
    """False marks a shortened/modified smoke run not comparable to other
    AgentX results (e.g. --agentx-num-dataset-entries)."""
    invalid_reasons: List[str] = Field(default_factory=list)
    """Wrapper-level validity explanations, complementing AIPerf's own."""

    # Normalized metrics (None = not reported)
    request_throughput: Optional[float] = None
    output_throughput: Optional[float] = None
    time_to_first_token_ms: Optional[float] = None
    inter_token_latency_ms: Optional[float] = None
    request_latency_ms: Optional[float] = None
    total_requests: Optional[int] = None
    successful_requests: Optional[int] = None
    error_requests: Optional[int] = None
    context_overflow_rate: Optional[float] = None

    artifact_dir: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump()


def _metric_value(metrics: Dict[str, Any], name: str, field: Optional[str] = None):
    """Fetch ``metrics[name][field or 'avg']`` without inventing zeros."""
    entry = metrics.get(name)
    if not isinstance(entry, dict):
        return None
    value = entry.get(field or 'avg')
    return value if isinstance(value, (int, float)) else None


def load_agentx_result(
    artifact_dir: str,
    aiperf_version: Optional[str] = None,
    aiperf_command: Optional[List[str]] = None,
    dataset_variant: str = 'full',
    random_seed: Optional[int] = None,
    model: Optional[str] = None,
    num_gpus: Optional[int] = None,
    smoke_reasons: Optional[List[str]] = None,
) -> AgentxResult:
    """Parse AIPerf exports under ``artifact_dir`` into an :class:`AgentxResult`.

    Prefers the aggregate export (multi-run) and falls back to the per-run
    export. Raw files are left untouched.
    """
    aggregate_path = os.path.join(artifact_dir, AGGREGATE_EXPORT)
    per_run_path = os.path.join(artifact_dir, PER_RUN_EXPORT)
    path = aggregate_path if os.path.isfile(aggregate_path) else per_run_path
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f'No AIPerf export found under {artifact_dir!r} '
            f'(looked for {AGGREGATE_EXPORT!r} and {PER_RUN_EXPORT!r})'
        )

    parsed = parse_profile_export(path)
    metadata, metrics = parsed['metadata'], parsed['metrics']

    invalid_reasons = list(smoke_reasons or [])
    reasons = metadata.get('submission_invalid_reasons') or []
    if isinstance(reasons, list):
        invalid_reasons.extend(str(r) for r in reasons)

    safe_command = _redact_command(aiperf_command) if aiperf_command else None

    return AgentxResult(
        aiperf_version=aiperf_version,
        aiperf_command=safe_command,
        dataset_variant=dataset_variant,
        dataset_repo=AGENTX_DATASET_REPOS.get(dataset_variant),
        random_seed=random_seed,
        model=model or metadata.get('model'),
        num_gpus=num_gpus,
        submission_valid=metadata.get('submission_valid'),
        submission_invalid_reasons=[str(r) for r in reasons] if isinstance(reasons, list) else [],
        canonical=not invalid_reasons if isinstance(metadata.get('submission_valid'), bool) else None,
        invalid_reasons=invalid_reasons,
        request_throughput=_metric_value(metrics, 'request_throughput'),
        output_throughput=_metric_value(metrics, 'output_token_throughput')
        or _metric_value(metrics, 'output_throughput'),
        time_to_first_token_ms=_ms(metrics, 'time_to_first_token', 'ttft'),
        inter_token_latency_ms=_ms(metrics, 'inter_token_latency', 'itl'),
        request_latency_ms=_ms(metrics, 'request_latency', 'e2el'),
        total_requests=_count(metrics, 'total_requests'),
        successful_requests=_count(metrics, 'successful_requests'),
        error_requests=_count(metrics, 'error_requests'),
        context_overflow_rate=_metric_value(metrics, 'context_overflow_rate'),
        artifact_dir=os.path.abspath(artifact_dir),
    )


def _redact_command(cmd: List[str]) -> List[str]:
    """Mask ``--api-key`` values so results can be logged safely."""
    redacted = list(cmd)
    for i, item in enumerate(redacted[:-1]):
        if item == '--api-key':
            redacted[i + 1] = '***'
    return redacted


def _ms(metrics: Dict[str, Any], *names: str) -> Optional[float]:
    for name in names:
        entry = metrics.get(name)
        if isinstance(entry, dict) and isinstance(entry.get('avg'), (int, float)):
            value = entry['avg']
            unit = str(entry.get('unit', 'ms')).lower()
            return value if unit in ('ms', 'millisecond', 'milliseconds') else value * 1000.0 \
                if unit in ('s', 'second', 'seconds') else value
    return None


def _count(metrics: Dict[str, Any], name: str) -> Optional[int]:
    value = metrics.get(name)
    if isinstance(value, dict):
        value = value.get('avg') or value.get('total') or value.get('count')
    return value if isinstance(value, int) else None
