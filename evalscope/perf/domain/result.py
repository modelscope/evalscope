from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field
from typing import Any, Dict, List, Optional

from evalscope.perf.config.models import PerfConfig
from evalscope.perf.config.resolve import ResolvedRunSpec
from evalscope.perf.metrics.definitions import MetricDefinition


class PercentileStats(BaseModel):
    p1: Optional[float] = None
    p5: Optional[float] = None
    p10: Optional[float] = None
    p25: Optional[float] = None
    p50: Optional[float] = None
    p75: Optional[float] = None
    p90: Optional[float] = None
    p95: Optional[float] = None
    p99: Optional[float] = None
    maximum: Optional[float] = None


class RunSummary(BaseModel):
    total: int = 0
    succeeded: int = 0
    failed: int = 0
    dropped: int = 0
    success_rate: float = 0.0
    duration_seconds: float = 0.0
    request_throughput: float = 0.0
    input_token_throughput: Optional[float] = None
    output_token_throughput: Optional[float] = None
    total_token_throughput: Optional[float] = None
    average_input_tokens: Optional[float] = None
    average_output_tokens: Optional[float] = None
    averages: Dict[str, float] = Field(default_factory=dict)


class TraceSummary(BaseModel):
    trace_count: int = 0
    averages: Dict[str, float] = Field(default_factory=dict)
    percentiles: Dict[str, PercentileStats] = Field(default_factory=dict)


class WorkloadSummary(BaseModel):
    overall: Dict[str, float] = Field(default_factory=dict)
    last_window: Dict[str, float] = Field(default_factory=dict)
    steady_state: Dict[str, float] = Field(default_factory=dict)


class ArtifactManifest(BaseModel):
    model_config = ConfigDict(frozen=True)

    root: str
    files: Dict[str, Optional[str]] = Field(default_factory=dict)


class RunResult(BaseModel):
    run_id: str
    run_spec: ResolvedRunSpec
    summary: RunSummary
    percentiles: Dict[str, PercentileStats] = Field(default_factory=dict)
    trace_summary: Optional[TraceSummary] = None
    workload_summary: Optional[WorkloadSummary] = None
    metric_definitions: Dict[str, MetricDefinition] = Field(default_factory=dict)
    artifacts: ArtifactManifest


class SLAEvaluation(BaseModel):
    load_value: float
    passed: Optional[bool] = None
    objective_value: Optional[float] = None
    run_ids: List[str] = Field(default_factory=list)


class SLAResult(BaseModel):
    variable: str
    best_value: Optional[float] = None
    evaluations: List[SLAEvaluation] = Field(default_factory=list)


class PerfSuiteResult(BaseModel):
    run_id: str
    suite_config: PerfConfig
    runs: List[RunResult]
    sla_result: Optional[SLAResult] = None
    artifacts: ArtifactManifest
