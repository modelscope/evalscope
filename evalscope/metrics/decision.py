"""Probability metrics and cross-sample aggregation for typed decisions."""

import json
import math
from collections import defaultdict

import numpy as np

from evalscope.api.metric import Aggregator, AggScore, Metric, SampleScore
from evalscope.api.metric.decision import NLL_EPSILON, DecisionObservation, decision_score
from evalscope.api.registry import register_aggregation, register_metric
from evalscope.metrics.aggregators.aggregators import Mean


class _ProbabilityMetric(Metric):
    operation = ''

    def apply(self, predictions: list[str], references: list[str]) -> list[float]:
        results = []
        for prediction, reference in zip(predictions, references):
            p = json.loads(prediction)
            y = json.loads(reference)
            if self.operation == 'nll':
                value = -math.log(max(p[int(y)], NLL_EPSILON))
            elif self.operation == 'brier':
                value = sum((v - float(i == int(y))) ** 2 for i, v in enumerate(p))
            elif self.operation == 'binary_brier':
                value = (p[1] - float(y)) ** 2
            elif self.operation == 'rps':
                value = sum((sum(p[: i + 1]) - float(int(y) <= i)) ** 2 for i in range(len(p) - 1)) / (len(p) - 1)
            else:
                value = abs(sum(i * v for i, v in enumerate(p)) - float(y))
            results.append(value)
        return results


@register_metric('decision_nll')
class DecisionNLL(_ProbabilityMetric):
    """Negative log likelihood, clipped at an explicitly recorded epsilon."""

    operation = 'nll'


@register_metric('decision_brier')
class DecisionBrier(_ProbabilityMetric):
    """Multiclass Brier score, summed over classes."""

    operation = 'brier'


@register_metric('decision_binary_brier')
class DecisionBinaryBrier(_ProbabilityMetric):
    """Binary Brier score for P(yes)."""

    operation = 'binary_brier'


@register_metric('decision_rps')
class DecisionRPS(_ProbabilityMetric):
    """Ranked probability score normalized by the number of level boundaries."""

    operation = 'rps'


@register_metric('decision_mae')
class DecisionMAE(_ProbabilityMetric):
    """Absolute error of the ordinal expected value."""

    operation = 'mae'


def ece(pairs: list[tuple[float, float]], bins: int = 10) -> float:
    """Equal-width expected calibration error; empty bins contribute nothing."""
    groups = defaultdict(list)
    for probability, outcome in pairs:
        groups[min(int(probability * bins), bins - 1)].append((probability, outcome))
    return sum(abs(sum(p - y for p, y in values)) for values in groups.values()) / len(pairs)


def average_ranks(values: list[float]) -> list[float]:
    """One-based average ranks, retaining ties for ROC AUC and Spearman correlation."""
    by_value = defaultdict(list)
    for index, value in enumerate(values):
        by_value[value].append(index)
    result = [0.0] * len(values)
    seen = 0
    for value in sorted(by_value):
        indices = by_value[value]
        rank = seen + (len(indices) + 1) / 2
        for index in indices:
            result[index] = rank
        seen += len(indices)
    return result


@register_aggregation('decision')
class DecisionAggregator(Aggregator):
    """Aggregate probabilities across samples, grouped by decision kind."""

    def __call__(self, scores: list[SampleScore]) -> list[AggScore]:
        groups = defaultdict(list)
        for sample in scores:
            data = (sample.score.metadata or {}).get('decision')
            if data:
                scale = len(data['labels']) if data['kind'] == 'score' else None
                groups[(data['kind'], scale)].append((sample, DecisionObservation.model_validate(data)))
        result = Mean()(scores)
        if len(groups) == 1:
            ((kind, scale),) = groups.keys()
            for aggregate in result:
                if aggregate.metric_name not in ('acc', 'accuracy'):
                    aggregate.dimensions = {'kind': kind, **({'levels': scale} if scale is not None else {})}
        if len(groups) > 1:
            # Different probability losses and Score scales must not share a denominator.
            result = [score for score in result if score.metric_name in ('acc', 'accuracy')]
        for (kind, scale), entries in groups.items():
            group_start = len(result)
            rows = [row for _, row in entries]
            ids = [s.sample_id for s, _ in entries]
            diagnostics = {}
            dimensions = {'kind': kind, **({'levels': scale} if scale is not None else {})}
            if len(groups) > 1:
                for aggregate in Mean()([sample for sample, _ in entries]):
                    aggregate.dimensions = dimensions
                    aggregate.aggregation = 'decision'
                    result.append(aggregate)

            def add(name: str, value: float, **extra_dimensions: str | float) -> None:
                if math.isfinite(float(value)):
                    result.append(
                        AggScore(
                            metric_name=name,
                            aggregation='decision',
                            score=float(value),
                            num=len(rows),
                            ids=ids,
                            dimensions={**dimensions, **extra_dimensions},
                            metadata=diagnostics,
                        )
                    )

            if not any(row.continuous for row in rows):
                calibration = [
                    (row.probabilities[row.labels.index('yes')], float(row.gold == 'yes'))
                    if kind == 'noul'
                    else (max(row.probabilities), row.correct)
                    for row in rows
                ]
                add('ece', ece(calibration), bins=10, mode='binary' if kind == 'noul' else 'top_label')
                if all(r.fixed_classes for r in rows) and len({tuple(sorted(r.labels)) for r in rows}) == 1:
                    f1s = []
                    for label in rows[0].labels:
                        tp = sum(r.predicted == label and str(r.gold) == label for r in rows)
                        fp = sum(r.predicted == label and str(r.gold) != label for r in rows)
                        fn = sum(r.predicted != label and str(r.gold) == label for r in rows)
                        f1s.append(2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0)
                    add('macro_f1', float(np.mean(f1s)))
            for basis in ['top_probability', 'provider_confidence']:
                eligible = [r for r in rows if basis != 'provider_confidence' or r.confidence is not None]
                if not eligible:
                    continue
                ordered = sorted(
                    eligible,
                    key=lambda r: -(r.confidence if basis == 'provider_confidence' else max(r.probabilities)),
                )
                losses = [
                    abs(r.expected - float(r.gold)) / (len(r.labels) - 1) if kind == 'score' else 1 - r.correct
                    for r in ordered
                ]
                risks = np.cumsum(losses) / np.arange(1, len(ordered) + 1)
                add(
                    'aurc',
                    float(np.mean(risks)),
                    confidence=basis,
                    risk='normalized_mae' if kind == 'score' else 'classification_error',
                )
                result[-1].num = len(eligible)
                result[-1].ids = [
                    s.sample_id for s, row in entries if basis != 'provider_confidence' or row.confidence is not None
                ]
                diagnostics[f'risk_coverage_{basis}'] = [
                    {'coverage': k / len(ordered), 'risk': risk} for k, risk in enumerate(risks, 1)
                ]
            if kind == 'noul':
                y = [int(r.gold == 'yes') for r in rows]
                predicted = [int(r.predicted == 'yes') for r in rows]
                tp = sum(a == b == 1 for a, b in zip(y, predicted))
                precision = tp / sum(predicted) if sum(predicted) else 0.0
                recall = tp / sum(y) if sum(y) else 0.0
                add('precision', precision)
                add('recall', recall)
                add('f1', 2 * precision * recall / (precision + recall) if precision + recall else 0.0)
                if len(set(y)) == 2:
                    p = [r.probabilities[r.labels.index('yes')] for r in rows]
                    ranks = average_ranks(p)
                    positives = sum(y)
                    add(
                        'auroc',
                        (sum(rank for rank, gold in zip(ranks, y) if gold) - positives * (positives + 1) / 2)
                        / (positives * (len(y) - positives)),
                    )
                else:
                    diagnostics.setdefault('unavailable', {})['auroc'] = 'Only one gold class.'
                if sum(y):
                    p = [r.probabilities[r.labels.index('yes')] for r in rows]
                    positives = sum(y)
                    tp = seen = 0
                    ap = 0.0
                    for probability in sorted(set(p), reverse=True):
                        indices = [i for i, value in enumerate(p) if value == probability]
                        added = sum(y[i] for i in indices)
                        tp += added
                        seen += len(indices)
                        ap += added / positives * tp / seen
                    add('auprc', ap)
                else:
                    diagnostics.setdefault('unavailable', {})['auprc'] = 'No positive gold observations.'
            if kind == 'score':
                golds = [float(r.gold) for r in rows]
                expected = [r.expected for r in rows]
                if len(rows) >= 2 and len(set(golds)) > 1 and len(set(expected)) > 1:
                    add('spearman', np.corrcoef(average_ranks(golds), average_ranks(expected))[0, 1])
                    add('pearson', np.corrcoef(golds, expected)[0, 1])
                else:
                    diagnostics['unavailable'] = {'correlations': 'Need nonconstant predictions and golds.'}
            for aggregate in result[group_start:]:
                aggregate.metadata = dict(diagnostics)
        families = defaultdict(list)
        for entries in groups.values():
            for _, row in entries:
                if row.family and not row.continuous:
                    families[row.family].append(row.correct)
        if families:
            result.append(
                AggScore(
                    metric_name='family_macro',
                    aggregation='decision',
                    score=float(np.mean([np.mean(values) for values in families.values()])),
                    num=sum(map(len, families.values())),
                    metadata={'families': {key: float(np.mean(values)) for key, values in families.items()}},
                )
            )
        return result
