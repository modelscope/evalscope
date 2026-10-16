"""Wire-format conversion for decision model APIs, shared with performance plugins."""

from typing import Any, Literal

from pydantic import BaseModel, Field

from evalscope.api.model.decision import (
    ANSWER_TOLERANCE,
    DecisionAnswer,
    DecisionInput,
    DecisionKind,
    Probability,
)

DecisionProtocol = Literal['systemone', 'autotrust']


def build_systemone_request(request: DecisionInput, model: str) -> dict[str, Any]:
    """Render the public System One wire format."""
    question = {'type': request.kind, 'instructions': request.instructions}
    if request.criteria is not None:
        question['criteria'] = request.criteria
    return {'model': model, 'state': request.state, 'questions': {'answer': question}}


def build_autotrust_request(request: DecisionInput) -> dict[str, Any]:
    """Render AutoTrust's native, fixed-six-level Score protocol."""
    if request.kind == 'score' and len(request.criteria) != 6:
        raise ValueError('AutoTrust /decide supports six Score levels (0-5) only; no implicit remapping.')
    instructions = request.instructions
    if request.kind == 'score':
        instructions += '\nLevels:\n' + '\n'.join(f'{i}: {v}' for i, v in enumerate(request.criteria))
    if request.kind == 'noul' and request.criteria:
        instructions += '\n' + '\n'.join(f'{k}: {v}' for k, v in request.criteria.items())
    payload = {'kind': request.kind, 'state': request.state, 'question': instructions, 'thinking': 'off'}
    if request.kind == 'choice':
        payload['options'] = _autotrust_options(request)
    return payload


def _autotrust_options(request: DecisionInput) -> list[str]:
    """Return wire options in the order used by AutoTrust's probability array."""
    if request.kind == 'choice':
        return list(request.criteria.values())
    if request.kind == 'noul':
        return ['false', 'true']
    return [str(i) for i in range(6)]


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


class _AutoTrustResponse(BaseModel):
    """Fields needed to map AutoTrust's array response onto the request labels."""

    kind: DecisionKind
    options: list[str] = Field(strict=True)
    probabilities: list[Probability] = Field(strict=True, min_length=2, max_length=255)
    choice_index: int = Field(strict=True, ge=0)
    choice: str


def _parse_systemone_answer(raw: dict, request: DecisionInput) -> DecisionAnswer:
    answers = raw.get('answers')
    if not isinstance(answers, dict) or set(answers) != {'answer'}:
        raise ValueError('System One must return exactly the requested answer question.')
    if not isinstance(answers['answer'], dict):
        raise ValueError('System One answer must be a JSON object.')
    data = dict(answers['answer'])
    probability = data.get('noul')
    if request.kind == 'noul' and 'probabilities' not in data and isinstance(probability, (int, float)):
        data['probabilities'] = {'no': 1 - probability, 'yes': probability}
    return DecisionAnswer.model_validate(data)


def _parse_autotrust_answer(raw: dict, request: DecisionInput) -> DecisionAnswer:
    response = _AutoTrustResponse.model_validate(raw)
    if response.kind != request.kind or raw.get('effective_kind', request.kind) != request.kind:
        raise ValueError('AutoTrust returned a different decision kind.')
    probabilities = response.probabilities
    if response.options != _autotrust_options(request) or len(probabilities) != len(request.labels):
        raise ValueError('AutoTrust options do not match the request.')
    index = response.choice_index
    if index >= len(response.options):
        raise ValueError('AutoTrust returned an invalid choice index.')
    if response.choice != response.options[index]:
        raise ValueError('AutoTrust choice and index disagree.')
    if probabilities[index] < max(probabilities) - ANSWER_TOLERANCE:
        raise ValueError('AutoTrust choice index must identify a highest-probability option.')
    data = {'type': request.kind, 'probabilities': dict(zip(request.labels, probabilities))}
    if request.kind == 'choice':
        data['choice'] = request.labels[index]
    elif request.kind == 'noul':
        data['noul'] = probabilities[1]
    else:
        data['score'] = sum(i * probability for i, probability in enumerate(probabilities))
        data['legend'] = dict(zip(request.labels, request.criteria))
    return DecisionAnswer.model_validate(data)


def parse_decision_response(
    raw: dict, request: DecisionInput, protocol: DecisionProtocol = 'systemone'
) -> DecisionAnswer:
    """Parse either wire format, then validate it against the original question."""
    if not isinstance(raw, dict) or not isinstance(raw.get('model'), str) or not raw['model']:
        raise ValueError('Decision response must include the resolved model name.')
    if protocol == 'systemone':
        answer = _parse_systemone_answer(raw, request)
    elif protocol == 'autotrust':
        answer = _parse_autotrust_answer(raw, request)
    else:
        raise ValueError(f'Unsupported decision protocol: {protocol}')
    if answer.type != request.kind or set(answer.probabilities) != set(request.labels):
        raise ValueError('Returned decision options do not match the request criteria.')
    if answer.type == 'score' and answer.legend != dict(zip(request.labels, request.criteria)):
        raise ValueError('Returned Score legend does not match the requested level descriptions.')
    return answer
