"""Provider-independent text decision contracts shared by evaluation and performance tests."""

import math
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator

from evalscope.constants import EvalType

DECISION_EVAL_TYPES = frozenset({EvalType.SYSTEMONE_API, EvalType.AUTOTRUST_API})
DecisionKind = Literal['choice', 'noul', 'score']
Probability = Annotated[float, Field(strict=True, ge=0, le=1, allow_inf_nan=False)]
PROBABILITY_TOLERANCE = 0.02
ANSWER_TOLERANCE = 1e-6


def _probability_sum_tolerance(values: list[float]) -> float:
    """Allow two-decimal rounding only when a unit-mass distribution could produce these values."""
    if not all(round(value, 2) == value for value in values):
        return PROBABILITY_TOLERANCE
    half_step = 0.005
    lower = sum(max(0.0, value - half_step) for value in values)
    upper = sum(min(1.0, value + half_step) for value in values)
    if lower <= 1 + 1e-12 and upper >= 1 - 1e-12:
        return max(PROBABILITY_TOLERANCE, len(values) * half_step)
    return PROBABILITY_TOLERANCE


class DecisionInput(BaseModel):
    """One typed question, without gold answers or scoring metadata."""

    model_config = ConfigDict(extra='forbid')
    kind: DecisionKind = 'choice'
    state: JsonValue
    instructions: str = Field(min_length=1)
    criteria: dict[str, str] | list[str] | None = None
    answer_prefix: Literal['ANSWER: ', '答案：'] = 'ANSWER: '

    @model_validator(mode='after')
    def validate_options(self) -> 'DecisionInput':
        if self.kind == 'choice':
            if not isinstance(self.criteria, dict) or not 2 <= len(self.criteria) <= 255:
                raise ValueError('Choice requires 2-255 named options.')
            if any(not key.strip() or not value.strip() for key, value in self.criteria.items()):
                raise ValueError('Choice options must not be empty.')
        elif self.kind == 'score':
            if not isinstance(self.criteria, list) or not 2 <= len(self.criteria) <= 10:
                raise ValueError('Score requires 2-10 ordered levels.')
            if any(not level.strip() for level in self.criteria):
                raise ValueError('Score level descriptions must not be empty.')
        elif self.criteria is not None:
            if not isinstance(self.criteria, dict) or set(self.criteria) - {'true', 'false'}:
                raise ValueError('Noul criteria may describe true and false only.')
        return self

    @property
    def labels(self) -> list[str]:
        """Return the canonical probability labels in request order."""
        if self.kind == 'choice':
            return list(self.criteria)
        if self.kind == 'noul':
            return ['no', 'yes']
        return [str(i) for i in range(len(self.criteria))]


class DecisionAnswer(BaseModel):
    """Canonical answer retaining the provider's unmodified probability values."""

    model_config = ConfigDict(extra='forbid')
    type: DecisionKind = 'choice'
    probabilities: dict[str, Probability] = Field(min_length=2, max_length=255)
    choice: str | None = None
    noul: Probability | None = None
    score: float | None = Field(default=None, allow_inf_nan=False, strict=True)
    legend: dict[str, str] | None = None
    confidence: Probability | None = None

    @model_validator(mode='after')
    def validate_distribution(self) -> 'DecisionAnswer':
        values = list(self.probabilities.values())
        tolerance = _probability_sum_tolerance(values)
        if not math.isclose(sum(values), 1, abs_tol=tolerance + 1e-12) or sum(values) <= 0:
            raise ValueError('Decision probabilities do not sum to 1 within their rounding tolerance.')
        if self.type == 'choice':
            if self.choice not in self.probabilities:
                raise ValueError('The selected choice is missing from probabilities.')
            if self.probabilities[self.choice] < max(values) - ANSWER_TOLERANCE:
                raise ValueError('The selected choice must have the highest probability.')
        elif self.type == 'noul':
            if set(self.probabilities) != {'no', 'yes'} or self.noul is None:
                raise ValueError('Noul must provide a probability of yes.')
            if not math.isclose(self.noul, self.probabilities['yes'], abs_tol=ANSWER_TOLERANCE):
                raise ValueError('Noul and yes probability disagree.')
        else:
            keys = [str(i) for i in range(len(self.probabilities))]
            if set(self.probabilities) != set(keys):
                raise ValueError('Score probability keys must be consecutive zero-based levels.')
            if self.legend is None or set(self.legend) != set(keys):
                raise ValueError('Score legend must match the probability levels.')
            expected = sum(int(k) * p for k, p in self.probabilities.items())
            if self.score is None or not 0 <= self.score <= len(keys) - 1:
                raise ValueError('Score must be inside the declared scale.')
            if not math.isclose(self.score, expected, abs_tol=tolerance * (len(keys) - 1) + ANSWER_TOLERANCE):
                raise ValueError('Score does not match the expected level.')
        return self
