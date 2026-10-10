"""Structured requests and answers for single-choice decision models."""

import math
from typing import Dict, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator

CHOICE_PROTOCOL_VERSION = 'v1.0'


class ChoiceQuestion(BaseModel):
    """One decision over a complete, ordered set of candidate answers."""

    model_config = ConfigDict(extra='forbid')

    type: Literal['choice'] = 'choice'
    instructions: str = Field(min_length=1)
    criteria: Dict[str, str] = Field(min_length=2, max_length=255)

    @model_validator(mode='after')
    def validate_options(self) -> 'ChoiceQuestion':
        if any(not key.strip() or not value.strip() for key, value in self.criteria.items()):
            raise ValueError('Choice option keys and descriptions must not be empty.')
        return self


class ChoiceRequest(BaseModel):
    """Task state and a single decision question, without gold labels."""

    model_config = ConfigDict(extra='forbid')

    state: JsonValue
    question: ChoiceQuestion


class ChoiceResult(BaseModel):
    """Selected option and the provider's probability distribution and confidence."""

    model_config = ConfigDict(extra='forbid')

    type: Literal['choice'] = 'choice'
    choice: str = Field(min_length=1)
    probabilities: Dict[str, float] = Field(min_length=2, max_length=255)
    confidence: Optional[float] = Field(default=None, ge=0, le=1, allow_inf_nan=False)

    @model_validator(mode='after')
    def validate_distribution(self) -> 'ChoiceResult':
        values = self.probabilities.values()
        if any(not math.isfinite(value) or not 0 <= value <= 1 for value in values):
            raise ValueError('Choice probabilities must be finite values between 0 and 1.')
        # Some System One providers round every probability to two decimal places.
        # Retain the raw values and allow only the corresponding rounding error.
        tolerance = max(0.01, len(self.probabilities) * 0.005) if all(round(v, 2) == v for v in values) else 0.01
        if not math.isclose(sum(values), 1, abs_tol=tolerance + 1e-12):
            raise ValueError('Choice probabilities do not sum to 1 within their rounding tolerance.')
        if self.choice not in self.probabilities:
            raise ValueError('The selected choice is missing from probabilities.')
        if self.probabilities[self.choice] < max(values) - 1e-6:
            raise ValueError('The selected choice must have the highest probability.')
        return self

    def validate_request(self, request: ChoiceRequest) -> None:
        """Reject missing, additional or unknown candidate answers."""
        if set(self.probabilities) != set(request.question.criteria):
            raise ValueError('Returned Choice options do not match the request criteria.')
