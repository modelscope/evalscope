"""Web envelopes reuse the persisted diagnostic contracts."""

from typing import List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, SecretStr

from evalscope.api.analysis import DiagnosisReport, SampleDiagnosis

from .common import ApiResponseModel


class DiagnosisResponse(ApiResponseModel):
    """Diagnostic artifact and freshness state."""

    status: Literal['ready', 'missing', 'stale', 'error']
    report: Optional[DiagnosisReport] = None


class DiagnosisRunRequest(BaseModel):
    """Options for analyzing one existing dataset report."""

    model_config = ConfigDict(extra='forbid')

    level: Literal['facts', 'diagnose'] = 'facts'
    model_id: Optional[str] = None
    api_url: Optional[str] = None
    api_key: Optional[SecretStr] = None
    max_diagnosed_samples: int = Field(default=50, ge=1, le=50, strict=True)
    scope: Literal['failed', 'unscored', 'risk', 'all'] = 'failed'
    include_trace: bool = True


class SampleDiagnosesResponse(ApiResponseModel):
    """Filtered sample observations, paged after matching."""

    status: Literal['ready', 'missing', 'stale', 'error']
    items: List[SampleDiagnosis]
    total: int
    page: int
    page_size: int
