import math
from typing import Annotated, Any, Dict, List, Optional, Union

from pydantic import AfterValidator, Field

from .common import ApiResponseModel
from .reports import MetricSemantics


def _finite_or_none(value: Optional[float]) -> Optional[float]:
    return value if value is not None and math.isfinite(value) else None


MetricValue = Annotated[Optional[float], AfterValidator(_finite_or_none)]
TableCell = Union[str, MetricValue]


class PerfRunSummary(ApiResponseModel):
    path: str
    model: str
    api_type: str
    dataset: str
    alias: Optional[str] = None
    num_runs: int
    total_requests: int
    success_rate: float
    best_rps: float
    best_latency: float
    avg_input_tokens: Optional[float] = None
    avg_output_tokens: Optional[float] = None
    is_embedding: bool
    has_html: bool
    timestamp: str
    provider: Optional[str] = None
    protocol: Optional[str] = None
    api_host: Optional[str] = None
    concurrency: Optional[List[int]] = None


class ListPerfRunsResponse(ApiResponseModel):
    runs: List[PerfRunSummary]
    total: int
    metric_semantics: Dict[str, MetricSemantics] = Field(default_factory=dict)


class PerfSummaryColumn(ApiResponseModel):
    key: str
    label: str
    semantics: Optional[MetricSemantics]


class PerfSummaryRow(ApiResponseModel):
    values: Dict[str, MetricValue]
    sample_counts: Dict[str, int]


class PerfDetailResponse(ApiResponseModel):
    path: str
    model: str
    api_type: str
    dataset: str
    alias: Optional[str] = None
    generated_at: str
    basic_info: Dict[str, str]
    configuration: Optional[Dict[str, str]] = None
    summary_columns: List[PerfSummaryColumn]
    summary_rows: List[PerfSummaryRow]
    total_requests: int
    best_config: Dict[str, str]
    recommendations: List[str]
    num_runs: int
    is_embedding: bool
    has_html: bool


class PerfRunItem(ApiResponseModel):
    dir_name: str
    name: str
    parallel: int
    number: int
    rate: Optional[float]
    total_requests: int
    succeed_requests: int
    success_rate: float
    num_requests: int
    has_requests: bool
    percentile_columns: List[str]
    percentile_rows: List[List[TableCell]]


class PerfRunsListResponse(ApiResponseModel):
    runs: List[PerfRunItem]
    total: int


class PerfRequestsResponse(ApiResponseModel):
    columns: List[str]
    rows: List[Dict[str, Any]]
    total: int
    page: int
    page_size: int
    has_db: bool


class DeletePerfRunResponse(ApiResponseModel):
    success: bool
    path: str
