from pydantic import BaseModel

from evalscope.api.metric import JudgeSummary
from evalscope.api.metric.semantics import MetricDisplayKind
from evalscope.report.report import ExecutionSummary

from .common import (
    ApiResponseModel,
    ConfigResponse,
    DataFrameResponse,
    LogResponse,
    ProgressResponse,
    TaskStatusResponse,
)
from .eval import BenchmarkEntry, BenchmarksResponse, EvalInvokeResponse
from .perf import (
    DeletePerfRunResponse,
    ListPerfRunsResponse,
    PerfDetailResponse,
    PerfRequestsResponse,
    PerfRunItem,
    PerfRunsListResponse,
    PerfRunSummary,
)
from .reports import (
    AnalysisResponse,
    ChatMessage,
    ContentBlock,
    DeleteReportResponse,
    JudgeAttempt,
    ListReportsResponse,
    LoadReportResponse,
    MetricIdentity,
    MetricSemantics,
    PercentileStats,
    PerfMetrics,
    PredictionRow,
    PredictionScore,
    PredictionsResponse,
    PredictionToolCall,
    ReportData,
    ReportMetric,
    ReportSummary,
    ValueRange,
)


class WebApiContracts(BaseModel):
    """Schema-only root that keeps every public Web API type reachable for code generation."""

    config_response: ConfigResponse
    data_frame_response: DataFrameResponse
    log_response: LogResponse
    task_status_response: TaskStatusResponse
    progress_response: ProgressResponse
    eval_invoke_response: EvalInvokeResponse
    benchmark_entry: BenchmarkEntry
    benchmarks_response: BenchmarksResponse
    report_data: ReportData
    report_summary: ReportSummary
    list_reports_response: ListReportsResponse
    load_report_response: LoadReportResponse
    percentile_stats: PercentileStats
    perf_metrics: PerfMetrics
    content_block: ContentBlock
    tool_call: PredictionToolCall
    chat_message: ChatMessage
    judge_attempt: JudgeAttempt
    judge_summary: JudgeSummary
    prediction_score: PredictionScore
    prediction_row: PredictionRow
    predictions_response: PredictionsResponse
    delete_report_response: DeleteReportResponse
    analysis_response: AnalysisResponse
    perf_run_summary: PerfRunSummary
    list_perf_runs_response: ListPerfRunsResponse
    perf_detail_response: PerfDetailResponse
    perf_run_item: PerfRunItem
    perf_runs_list_response: PerfRunsListResponse
    perf_requests_response: PerfRequestsResponse
    delete_perf_run_response: DeletePerfRunResponse
    metric_identity: MetricIdentity
    metric_display_kind: MetricDisplayKind
    metric_semantics: MetricSemantics
    value_range: ValueRange
    report_metric: ReportMetric
    execution_summary: ExecutionSummary


__all__ = [
    'AnalysisResponse',
    'ApiResponseModel',
    'BenchmarkEntry',
    'BenchmarksResponse',
    'ConfigResponse',
    'DataFrameResponse',
    'DeletePerfRunResponse',
    'DeleteReportResponse',
    'EvalInvokeResponse',
    'ListPerfRunsResponse',
    'ListReportsResponse',
    'LoadReportResponse',
    'LogResponse',
    'PerfDetailResponse',
    'PerfRequestsResponse',
    'PerfRunsListResponse',
    'PredictionsResponse',
    'ProgressResponse',
    'TaskStatusResponse',
    'WebApiContracts',
]
