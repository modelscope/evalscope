"""Mathematical comparison results and scoring failures."""

from pydantic import BaseModel, ConfigDict

from evalscope.api.metric import MetricUnavailableError


class MathResult(BaseModel):
    """Parsed-answer validity and the mathematical comparison outcome."""

    model_config = ConfigDict(extra='forbid')
    extracted: str = ''
    matched: bool = False
    prediction_valid: bool = False
    reference_valid: bool = False
    reason: str = ''


class MathEvaluationError(MetricUnavailableError):
    """Scoring is unavailable rather than a valid incorrect prediction."""


class InvalidMathReference(MathEvaluationError):
    """A reference cannot be parsed or violates the required benchmark type."""
