"""结构化决策的跨样本概率指标聚合。"""

import math
from collections import defaultdict

import numpy as np

from evalscope.api.metric import Aggregator, AggScore, SampleScore
from evalscope.api.metric.decision import DecisionObservation
from evalscope.api.registry import register_aggregation
from evalscope.metrics.utils.functions import auroc, average_ranks, ece

from .aggregators import Mean


@register_aggregation('decision')
class DecisionAggregator(Aggregator):
    """按决策类型分组，聚合跨样本的概率与校准指标。"""

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
            # 不同概率损失和评分尺度不能共用分母。
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
                class_maps = [r.class_labels or {label: label for label in r.labels} for r in rows]
                if all(r.fixed_classes for r in rows) and len({tuple(sorted(m.values())) for m in class_maps}) == 1:
                    f1s = []
                    predictions = [m[r.predicted] for m, r in zip(class_maps, rows)]
                    golds = [m[str(r.gold)] for m, r in zip(class_maps, rows)]
                    for label in class_maps[0].values():
                        tp = sum(p == g == label for p, g in zip(predictions, golds))
                        fp = sum(p == label and g != label for p, g in zip(predictions, golds))
                        fn = sum(p != label and g == label for p, g in zip(predictions, golds))
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
                confidences = [
                    r.confidence if basis == 'provider_confidence' else max(r.probabilities) for r in ordered
                ]
                cumulative = np.cumsum(losses)
                curve = []
                area = 0.0
                previous = 0
                for end in range(1, len(ordered) + 1):
                    if end < len(ordered) and confidences[end - 1] == confidences[end]:
                        continue
                    risk = float(cumulative[end - 1] / end)
                    curve.append({'coverage': end / len(ordered), 'risk': risk})
                    area += (end - previous) / len(ordered) * risk
                    previous = end
                add(
                    'aurc',
                    area,
                    confidence=basis,
                    risk='normalized_mae' if kind == 'score' else 'classification_error',
                )
                result[-1].num = len(eligible)
                result[-1].ids = [
                    s.sample_id for s, row in entries if basis != 'provider_confidence' or row.confidence is not None
                ]
                diagnostics[f'risk_coverage_{basis}'] = curve
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
                    add('auroc', auroc([r.probabilities[r.labels.index('yes')] for r in rows], y))
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
