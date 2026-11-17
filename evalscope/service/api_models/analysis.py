"""Web envelopes reuse the persisted diagnostic contracts."""

from typing import Dict, List, Literal, Optional

from pydantic import Field

from evalscope.api.analysis import DiagnosisReport, SampleDiagnosis

from .common import ApiResponseModel


class DiagnosisIssue(ApiResponseModel):
    """Observed issue groups; identities are unique within each group and groups may overlap."""

    failure_type: str
    failed: int = 0
    unscored: int = 0
    tools: Dict[str, int] = Field(default_factory=dict)


class DiagnosisResponse(ApiResponseModel):
    """Read-only diagnostic artifact and freshness state."""

    status: Literal['ready', 'missing', 'stale', 'error']
    report: Optional[DiagnosisReport] = None
    issues: List[DiagnosisIssue] = Field(default_factory=list)
    risk_samples: int = 0
    passed_observations: int = 0


class SampleDiagnosesResponse(ApiResponseModel):
    """Filtered sample observations, paged after matching."""

    status: Literal['ready', 'missing', 'stale', 'error']
    items: List[SampleDiagnosis]
    total: int
    page: int
    page_size: int
