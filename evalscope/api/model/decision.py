"""Validated text decisions shared by model transports, scoring and performance plugins."""

import math
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator

DECISION_EVAL_TYPES = frozenset({'systemone_api', 'autotrust_api'})


def parse_decision_usage(raw: dict) -> tuple[int, int]:
    """Read optional usage in either protocol, rejecting malformed token counts."""
    usage = raw.get('usage')
    if usage is None:
        usage = {}
    if not isinstance(usage, dict):
        raise ValueError('Decision usage must be a JSON object.')
    counts = (
        usage.get('input_tokens', usage.get('prompt_tokens', 0)),
        usage.get('output_tokens', usage.get('completion_tokens', 0)),
    )
    if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in counts):
        raise ValueError('Decision token counts must be nonnegative integers.')
    return counts


class DecisionInput(BaseModel):
    """One typed question, without gold answers or scoring metadata."""

    model_config = ConfigDict(extra='forbid')
    kind: Literal['choice', 'noul', 'score'] = 'choice'
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

    def systemone_payload(self, model: str) -> dict[str, Any]:
        """Render the public System One wire format."""
        question = {'type': self.kind, 'instructions': self.instructions}
        if self.criteria is not None:
            question['criteria'] = self.criteria
        return {'model': model, 'state': self.state, 'questions': {'answer': question}}

    def autotrust_payload(self) -> dict[str, Any]:
        """Render AutoTrust's native, fixed-six-level Score protocol."""
        if self.kind == 'score' and len(self.criteria) != 6:
            raise ValueError('AutoTrust /decide supports six Score levels (0-5) only; no implicit remapping.')
        instructions = self.instructions
        if self.kind == 'score':
            instructions += '\nLevels:\n' + '\n'.join(f'{i}: {v}' for i, v in enumerate(self.criteria))
        if self.kind == 'noul' and self.criteria:
            instructions += '\n' + '\n'.join(f'{k}: {v}' for k, v in self.criteria.items())
        payload = {'kind': self.kind, 'state': self.state, 'question': instructions, 'thinking': 'off'}
        if self.kind == 'choice':
            payload['options'] = list(self.criteria.values())
        return payload


class DecisionAnswer(BaseModel):
    """Canonical answer retaining the provider's unmodified probability values."""

    model_config = ConfigDict(extra='forbid')
    type: Literal['choice', 'noul', 'score'] = 'choice'
    probabilities: dict[str, float] = Field(min_length=2, max_length=255)
    choice: str | None = None
    noul: float | None = Field(default=None, ge=0, le=1, allow_inf_nan=False)
    score: float | None = Field(default=None, allow_inf_nan=False)
    legend: dict[str, str] | None = None
    confidence: float | None = Field(default=None, ge=0, le=1, allow_inf_nan=False)

    @model_validator(mode='after')
    def validate_distribution(self) -> 'DecisionAnswer':
        values = self.probabilities.values()
        if any(not math.isfinite(v) or not 0 <= v <= 1 for v in values):
            raise ValueError('Decision probabilities must be finite and between 0 and 1.')
        tolerance = max(0.01, len(self.probabilities) * 0.005) if all(round(v, 2) == v for v in values) else 0.01
        if not math.isclose(sum(values), 1, abs_tol=tolerance + 1e-12) or sum(values) <= 0:
            raise ValueError('Decision probabilities do not sum to 1 within their rounding tolerance.')
        if self.type == 'choice':
            if self.choice not in self.probabilities:
                raise ValueError('The selected choice is missing from probabilities.')
            if self.probabilities[self.choice] < max(values) - 1e-6:
                raise ValueError('The selected choice must have the highest probability.')
        elif self.type == 'noul':
            if set(self.probabilities) != {'no', 'yes'} or self.noul is None:
                raise ValueError('Noul must provide a probability of yes.')
            if not math.isclose(self.noul, self.probabilities['yes'], abs_tol=1e-6):
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
            if not math.isclose(self.score, expected, abs_tol=tolerance * (len(keys) - 1) + 1e-6):
                raise ValueError('Score does not match the expected level.')
        return self


def parse_decision_response(raw: dict, request: DecisionInput, protocol: str = 'systemone') -> DecisionAnswer:
    """Validate a provider result against the exact question that was sent."""
    if not isinstance(raw, dict) or not isinstance(raw.get('model'), str) or not raw['model']:
        raise ValueError('Decision response must include the resolved model name.')
    if protocol == 'systemone':
        if not isinstance(raw.get('answers'), dict) or set(raw['answers']) != {'answer'}:
            raise ValueError('System One must return exactly the requested answer question.')
        data = dict(raw['answers']['answer'])
        if request.kind == 'noul':
            p = data.get('noul')
            if 'probabilities' not in data and isinstance(p, (float, int)):
                data['probabilities'] = {'no': 1 - p, 'yes': p}
    else:
        if raw.get('kind') != request.kind or raw.get('effective_kind', request.kind) != request.kind:
            raise ValueError('AutoTrust returned a different decision kind.')
        options = (
            list(request.criteria.values())
            if request.kind == 'choice'
            else ['false', 'true']
            if request.kind == 'noul'
            else [str(i) for i in range(6)]
        )
        probs = raw.get('probabilities')
        if raw.get('options') != options or not isinstance(probs, list) or len(probs) != len(options):
            raise ValueError('AutoTrust options do not match the request.')
        index = raw.get('choice_index')
        if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < len(options):
            raise ValueError('AutoTrust returned an invalid choice index.')
        if raw.get('choice') != options[index]:
            raise ValueError('AutoTrust choice and index disagree.')
        if any(not isinstance(p, (int, float)) or not math.isfinite(p) for p in probs):
            raise ValueError('AutoTrust probabilities must be finite numbers.')
        if probs[index] < max(probs) - 1e-6:
            raise ValueError('AutoTrust choice index must identify a highest-probability option.')
        data = {'type': request.kind, 'probabilities': dict(zip(request.labels, probs))}
        if request.kind == 'choice':
            data['choice'] = request.labels[index]
        elif request.kind == 'noul':
            data['noul'] = probs[1]
        else:
            data['score'] = sum(i * p for i, p in enumerate(probs))
            data['legend'] = dict(zip(request.labels, request.criteria))
    answer = DecisionAnswer.model_validate(data)
    if answer.type != request.kind or set(answer.probabilities) != set(request.labels):
        raise ValueError('Returned decision options do not match the request criteria.')
    if answer.type == 'score' and answer.legend != dict(zip(request.labels, request.criteria)):
        raise ValueError('Returned Score legend does not match the requested level descriptions.')
    return answer
