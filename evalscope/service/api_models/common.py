from typing import Any, Dict, Optional

from pydantic import BaseModel, ConfigDict


class ApiResponseModel(BaseModel):
    """Strict base model for JSON responses consumed by the bundled Web UI."""

    model_config = ConfigDict(extra='forbid', populate_by_name=True)


class ConfigResponse(ApiResponseModel):
    outputs_root: str


class DataFrameResponse(ApiResponseModel):
    columns: list[str]
    data: list[Dict[str, Any]]


class LogResponse(ApiResponseModel):
    text: str
    head_line: int
    tail_line: int
    total_lines: int


class TaskStatusResponse(ApiResponseModel):
    status: str
    task_id: str


class ProgressResponse(ApiResponseModel):
    """Progress tracker payload with stable fields and tracker-specific metadata."""

    model_config = ConfigDict(extra='allow')

    percent: float
    current_step: Optional[str] = None
    status: Optional[str] = None
