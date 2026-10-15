"""Typed decision observations and registry-based sample scoring."""

import json
import math

from pydantic import BaseModel

from evalscope.api.metric import Score
from evalscope.api.model.decision import DecisionAnswer
from evalscope.api.registry import get_metric

NLL_EPSILON = 1e-12


class DecisionObservation(BaseModel):
    """Validated scoring inputs retained for reproducible cross-sample metrics."""

    kind: str
    labels: list[str]
    probabilities: list[float]
    raw_probabilities: dict[str, float]
    gold: str | float
    predicted: str
    correct: float
    expected: float | None = None
    confidence: float | None = None
    threshold: float | None = 0.5
    continuous: bool = False
    family: str = ''
    tier: str = ''
    fixed_classes: bool = False
    class_labels: dict[str, str] | None = None
    noul_rule: str = 'threshold'


def decision_score(
    answer: DecisionAnswer,
    gold: str | float,
    labels: list[str],
    *,
    continuous: bool = False,
    threshold: float = 0.5,
    family: str = '',
    tier: str = '',
    fixed_classes: bool = False,
    class_labels: dict[str, str] | None = None,
    normalize: bool = True,
    noul_use_argmax: bool = False,
) -> Score:
    """Score a decision while retaining raw probabilities separately from scoring probabilities."""
    if not 0 <= threshold <= 1:
        raise ValueError('Decision threshold must be between zero and one.')
    total = sum(answer.probabilities.values())
    probs = [answer.probabilities[label] / (total if normalize else 1) for label in labels]
    best = max(sorted(labels), key=lambda label: probs[labels.index(label)])
    if answer.type == 'noul' and not noul_use_argmax:
        best = 'yes' if probs[labels.index('yes')] >= threshold else 'no'
    expected = sum(i * p for i, p in enumerate(probs)) if answer.type == 'score' else None
    correct = float(best == str(gold)) if not continuous else 0.0
    observation = DecisionObservation(
        kind=answer.type,
        labels=labels,
        probabilities=probs,
        raw_probabilities=answer.probabilities,
        gold=gold,
        predicted=best,
        correct=correct,
        expected=expected,
        confidence=answer.confidence,
        threshold=None if noul_use_argmax else threshold,
        continuous=continuous,
        family=family,
        tier=tier,
        fixed_classes=fixed_classes,
        class_labels=class_labels,
        noul_rule='argmax' if noul_use_argmax else 'threshold',
    )
    value = {} if continuous else {'acc': correct}
    encoded = json.dumps(probs)
    if answer.type == 'score':
        value['mae'] = get_metric('decision_mae')()(encoded, json.dumps(float(gold)))
        if not continuous:
            value['nrps'] = get_metric('decision_rps')()(encoded, json.dumps(labels.index(str(gold))))
    if not continuous:
        index = labels.index(str(gold))
        value['nll'] = get_metric('decision_nll')()(encoded, json.dumps(index))
        metric = 'decision_binary_brier' if answer.type == 'noul' else 'decision_brier'
        value['brier'] = get_metric(metric)()(encoded, json.dumps(index))
    return Score(
        value=value,
        main_score_name='mae' if continuous else 'acc',
        extracted_prediction=str(expected) if continuous else best,
        metadata={
            'decision': observation.model_dump(),
            'probability_sum': total,
            'probabilities_normalized_for_scoring': normalize and not math.isclose(total, 1, abs_tol=1e-12),
            'nll_epsilon': NLL_EPSILON,
        },
    )
