from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class SLAProbe(BaseModel):
    """One tested pressure value and the evidence used for its SLA decision."""

    value: int
    averaged_result: Dict[str, Any]
    total_requests: int
    succeeded_requests: int
    success_rate: float
    request_gate_passed: bool
    valid: bool
    reasons: List[str] = Field(default_factory=list)
    metric_values: Dict[str, float] = Field(default_factory=dict)
    group_passes: Dict[str, bool] = Field(default_factory=dict)


class SLASelection(BaseModel):
    """Best tested value for one constraint group or optimization objective."""

    criteria: Dict[str, str]
    mode: Literal['constraint', 'max', 'min']
    selected_value: Optional[int] = None
    observed_metric: Optional[float] = None
    status: Literal['best_observed', 'none']
    reason: str
    assumption: str
