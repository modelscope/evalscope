"""Wire contracts for isolated Math-Verify execution; math objects never leave the child."""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

from evalscope.api.metric import MetricUnavailableError


class MathRequest(BaseModel):
    """One extraction or comparison with benchmark-specific numeric requirements."""

    model_config = ConfigDict(extra='forbid')
    operation: Literal['extract', 'compare', 'boxed', 'number', 'render'] = 'compare'
    prediction: str
    reference: str = ''
    prediction_mode: Literal['output', 'fragment'] = 'fragment'
    absolute_tolerance: list[Annotated[float, Field(ge=0, allow_inf_nan=False)]] | None = Field(
        default=None, min_length=1
    )
    relative_tolerance: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    multiple_answers: bool = False
    integer_only: bool = False
    numeric_reference: bool = False


class MathResult(BaseModel):
    """Only display text, booleans and failure information cross the process boundary."""

    model_config = ConfigDict(extra='forbid')
    extracted: str = ''
    number: float | None = None
    parts: list[str] = Field(default_factory=list)
    matched: bool = False
    prediction_valid: bool = False
    reference_valid: bool = False
    reason: str = ''
    error: str | None = None


class MathEvaluationError(MetricUnavailableError):
    """Scoring is unavailable rather than a valid incorrect prediction."""


class InvalidMathReference(MathEvaluationError):
    """A reference cannot be parsed or violates the required benchmark type."""
