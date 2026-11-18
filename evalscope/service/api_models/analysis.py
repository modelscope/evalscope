"""Web envelopes reuse the persisted diagnostic contracts."""

from typing import List, Literal, Optional

from evalscope.api.analysis import DiagnosisReport, SampleDiagnosis

from .common import ApiResponseModel


class DiagnosisResponse(ApiResponseModel):
    """Read-only diagnostic artifact and freshness state."""

    status: Literal['ready', 'missing', 'stale', 'error']
    report: Optional[DiagnosisReport] = None


class SampleDiagnosesResponse(ApiResponseModel):
    """Filtered sample observations, paged after matching."""

    status: Literal['ready', 'missing', 'stale', 'error']
    items: List[SampleDiagnosis]
    total: int
    page: int
    page_size: int
