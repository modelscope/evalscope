from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import ConfigDict, Field

from evalscope.api.agent.trace import EventType
from evalscope.api.metric import JudgeSummary
from evalscope.api.metric.semantics import MetricDirection, MetricDisplayKind, MetricKind
from evalscope.constants import ScoreStatus
from evalscope.report.report import ExecutionSummary

from .common import ApiResponseModel


class MetricIdentity(ApiResponseModel):
    name: str
    aggregation: str
    dimensions: Dict[str, Union[str, int, float, bool]]


class ValueRange(ApiResponseModel):
    min: float
    max: float


class MetricSemantics(ApiResponseModel):
    semantic_id: str
    metric_name: str
    display_name: Optional[str] = None
    kind: MetricKind
    direction: MetricDirection
    raw_unit: Optional[str] = None
    value_range: Optional[ValueRange] = None
    display_kind: MetricDisplayKind
    display_multiplier: Optional[float] = None
    display_unit: Optional[str] = None
    display_precision: int


class ReportSubset(ApiResponseModel):
    name: str
    score: float
    num: int
    is_aggregate: bool = False


class ReportCategory(ApiResponseModel):
    name: List[str]
    num: int
    score: float
    macro_score: float = 0.0
    subsets: List[ReportSubset]


class ReportMetric(ApiResponseModel):
    identity: MetricIdentity
    legacy_name: Optional[str] = None
    num: int
    score: float
    macro_score: float = 0.0
    categories: List[ReportCategory]
    semantics: MetricSemantics


class PercentileStats(ApiResponseModel):
    mean: float
    std: Optional[float]
    min: float
    percentile_25: float = Field(alias='25%')
    percentile_50: float = Field(alias='50%')
    percentile_75: float = Field(alias='75%')
    percentile_90: float = Field(alias='90%')
    percentile_99: float = Field(alias='99%')
    max: float


class ThroughputSummary(ApiResponseModel):
    avg_output_tps: float
    avg_req_ps: float


class UsageSummary(ApiResponseModel):
    input_tokens: PercentileStats
    output_tokens: PercentileStats
    total_tokens: PercentileStats
    total_input_tokens: Optional[float] = None
    total_output_tokens: Optional[float] = None
    total_tokens_count: Optional[float] = None


class PerfMetricsSummary(ApiResponseModel):
    n_samples: int
    latency: PercentileStats
    throughput: ThroughputSummary
    usage: UsageSummary
    ttft: Optional[PercentileStats] = None
    tpot: Optional[PercentileStats] = None


class PerfCoverage(ApiResponseModel):
    requests_with_metrics: int
    total_requests: int


class PerfMetrics(ApiResponseModel):
    summary: Optional[PerfMetricsSummary] = None
    metric_semantics: Dict[str, MetricSemantics] = Field(default_factory=dict)
    coverage: Optional[PerfCoverage] = None


class ReportData(ApiResponseModel):
    """Canonical report payload emitted by ``Report.to_dict``."""

    schema_version: Literal[2]
    name: str
    dataset_name: str
    dataset_pretty_name: str = ''
    dataset_description: str = ''
    model_name: str
    metrics: List[ReportMetric]
    analysis: str
    perf_metrics: Optional[PerfMetrics] = None
    primary_metric_identity: Optional[MetricIdentity] = None
    primary_metric_unavailable_reason: Optional[str] = None
    judge_summary: Optional[JudgeSummary] = None
    execution_summary: Optional[ExecutionSummary] = None
    num: int = 0


class PrimaryMetricRef(ApiResponseModel):
    dataset_name: str
    dataset_pretty_name: str = ''
    identity: MetricIdentity
    score: float
    semantics: MetricSemantics


class ReportSummary(ApiResponseModel):
    run_id: str
    model_id: str
    model_name: str
    dataset_name: str
    dataset_pretty_name: str = ''
    num_samples: int
    timestamp: str
    primary_metrics: List[PrimaryMetricRef]


class ReportFilters(ApiResponseModel):
    available_models: List[str]
    available_datasets: List[str]


class ListReportsResponse(ApiResponseModel):
    reports: List[ReportSummary]
    total: int
    page: int
    page_size: int
    filters: ReportFilters


class ReportGroup(ApiResponseModel):
    """One row per model in a `group_by=model` listing - display-only rollup, never a merged report."""

    model_name: str
    dataset_name: str
    timestamp: str
    report_count: int
    dataset_count: int
    num_samples: int
    refs: List[str]
    children: List[ReportSummary]


class ListReportsGroupedResponse(ApiResponseModel):
    reports: List[ReportGroup]
    total: int
    page: int
    page_size: int
    filters: ReportFilters


class LoadReportResponse(ApiResponseModel):
    report_list: List[ReportData]
    datasets: List[str]
    task_config: Dict[str, Any]


class SamplePerfMetrics(ApiResponseModel):
    latency: float
    ttft: Optional[float] = None
    tpot: Optional[float] = None
    input_tokens: int
    output_tokens: int


class ContentBlock(ApiResponseModel):
    model_config = ConfigDict(extra='allow')

    type: Literal['text', 'reasoning', 'image', 'audio', 'video', 'data']
    text: Optional[str] = None
    reasoning: Optional[str] = None
    reasoning_tokens: Optional[int] = None
    image: Optional[str] = None
    audio: Optional[str] = None
    video: Optional[str] = None
    format: Optional[str] = None
    detail: Optional[str] = None
    data: Optional[Dict[str, Any]] = None


class PredictionToolCall(ApiResponseModel):
    id: str
    function: str
    arguments: Dict[str, Any]


class ToolMessageError(ApiResponseModel):
    type: Optional[str] = None
    message: str


class ChatMessage(ApiResponseModel):
    id: Optional[str] = None
    role: Literal['system', 'user', 'assistant', 'tool']
    content: Union[str, List[ContentBlock]]
    source: Optional[Literal['input', 'generate']] = None
    metadata: Optional[Dict[str, Any]] = None
    perf_metrics: Optional[SamplePerfMetrics] = None
    tool_calls: Optional[List[PredictionToolCall]] = None
    model: Optional[str] = None
    tool_call_id: Optional[Union[str, List[str]]] = None
    function: Optional[str] = None
    error: Optional[ToolMessageError] = None


class AgentTraceEvent(ApiResponseModel):
    step: int
    timestamp: float
    type: EventType
    message_id: Optional[str] = None
    latency_ms: Optional[float] = None
    token_usage: Optional[Dict[str, int]] = None
    payload: Dict[str, Any]


class AgentTrace(ApiResponseModel):
    strategy: Optional[str] = None
    environment: Optional[str] = None
    max_steps: int
    events: List[AgentTraceEvent]


class JudgeAttempt(ApiResponseModel):
    status: ScoreStatus
    case_id: str
    judge_id: str
    repeat_id: int = 0
    placement: Literal['original', 'swapped'] = 'original'
    messages: List[Any] = Field(default_factory=list)
    model_output: Any = None
    raw_response: Optional[str] = None
    parsed_value: Any = None
    error: Optional[str] = None
    latency: Optional[float] = None


class PredictionMetadata(ApiResponseModel):
    model_config = ConfigDict(extra='allow')

    judge_attempts: Optional[List[JudgeAttempt]] = None
    judge_skipped: Optional[bool] = None
    judge_skip_reason: Optional[str] = None


class PredictionScore(ApiResponseModel):
    """Score fields rendered by the Web UI; metadata remains benchmark-defined."""

    model_config = ConfigDict(extra='allow')

    value: Dict[str, Union[int, float, bool]] = Field(default_factory=dict)
    status: ScoreStatus = ScoreStatus.SUCCESS
    judge_summary: Optional[JudgeSummary] = None
    explanation: Optional[str] = None
    main_score_name: Optional[str] = None
    metadata: Optional[PredictionMetadata] = Field(default_factory=PredictionMetadata)


class PredictionRow(ApiResponseModel):
    index: str = Field(alias='Index')
    input: str = Field(alias='Input')
    metadata: Any = Field(alias='Metadata')
    generated: str = Field(alias='Generated')
    gold: str = Field(alias='Gold')
    prediction: str = Field(alias='Pred')
    score: PredictionScore = Field(alias='Score')
    normalized_score: Optional[float] = Field(alias='NScore')
    status: Optional[str] = Field(default=None, alias='Status')
    perf_metrics: Optional[SamplePerfMetrics] = Field(default=None, alias='PerfMetrics')
    messages: Optional[List[ChatMessage]] = Field(default=None, alias='Messages')
    agent_trace: Optional[AgentTrace] = Field(default=None, alias='AgentTrace')


class PredictionsResponse(ApiResponseModel):
    predictions: List[PredictionRow]


class DeleteReportResponse(ApiResponseModel):
    success: bool
    run_id: str
    model_id: str


class AnalysisResponse(ApiResponseModel):
    analysis: str
