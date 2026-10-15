# Adapted from jgeuter/OpenDecisions score.py (MIT License).
# https://huggingface.co/datasets/jgeuter/OpenDecisions
# Copyright OpenDecisions authors; retained scoring semantics.
# Reference revision: 5026a13ad6e5a8606810a3a5c4e410d8d6638e71
# score.py SHA-256: 06994fe9fffca7819c34b3dd985ab4fdcdd00fa115d8e37a98076377dcb0d03e

from __future__ import annotations

import math
from collections import defaultdict

TIER_WEIGHTS = {'easy': 0.1, 'standard': 0.2, 'judge': 0.3, 'hard': 0.4}
TYPES = ('choice', 'noul', 'score')


# ------------------------------------------------------------------ per item


def ece(pairs: list[tuple[float, float]], bins: int = 10) -> float | None:
    """Expected calibration error of (forecast, outcome) pairs; equal-width bins, empty bins absent."""
    if not pairs:
        return None
    sums = defaultdict(lambda: [0, 0.0, 0.0])
    for p, y in pairs:
        p = min(max(float(p), 0.0), 1.0)
        b = sums[min(int(p * bins), bins - 1)]
        b[0] += 1
        b[1] += p
        b[2] += float(y)
    return sum(n / len(pairs) * abs(o / n - f / n) for n, f, o in sums.values())


def nrps(probs: list[float], gold: int) -> float:
    """Normalized ranked probability score (0 is perfect)."""
    k, cum, total = len(probs), 0.0, 0.0
    for level in range(k - 1):
        cum += probs[level]
        total += (cum - (1.0 if gold <= level else 0.0)) ** 2
    return total / (k - 1)


def item_outcome(task: dict, probs: dict | None) -> dict:
    """Outcome of one item; probs is None for a missing prediction (wrong, no forecast)."""
    qtype, labels = task['question']['type'], task['labels']
    out = {'id': task['id'], 'type': qtype, 'tier': task['tier'], 'k': len(labels), 'valid': probs is not None}
    if qtype == 'choice':
        out['chance'] = 1.0 / len(labels)
        if probs is None:
            out.update(correct=False, conf=None)
        else:
            best = max(sorted(labels), key=lambda lab: probs[lab])  # ties: the smallest label
            out.update(correct=best == task['expected'], conf=probs[best])
    elif qtype == 'noul':
        y = 1.0 if task['expected'] == 'yes' else 0.0
        if probs is None:
            out.update(correct=False, p_yes=None, y=y)
        else:
            p = probs['yes']
            out.update(correct=(p >= 0.5) == (y == 1.0), p_yes=p, y=y)  # the more likely side; no abstention
    else:
        k, gold = len(labels), int(task['expected'])
        out['nmae_chance'] = sum(abs(level - gold) for level in range(k)) / k / (k - 1)  # random level
        out['nmae_chance_mid'] = abs((k - 1) / 2 - gold) / (k - 1)  # mid-scale guess
        if probs is None:
            out.update(nmae=max(gold, k - 1 - gold) / (k - 1), conf=None, correct=False)  # the worst level
        else:
            vector = [probs[str(level)] for level in range(k)]
            ev = sum(level * p for level, p in enumerate(vector))
            best = max(range(k), key=lambda level: (vector[level], -level))
            out.update(nmae=abs(ev - gold) / (k - 1), conf=vector[best], correct=best == gold, nrps=nrps(vector, gold))
    return out


# ------------------------------------------------------------------ aggregation


def _weighted(values: dict, weights: dict) -> float | None:
    present = {k: v for k, v in values.items() if v is not None}
    total = sum(weights[k] for k in present)
    return sum(weights[k] * v for k, v in present.items()) / total if total else None


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def competence(outs: list[dict]) -> float | None:
    """Chance-corrected competence of items of one type and tier (0 = chance, 100 = perfect)."""
    if not outs:
        return None
    qtype = outs[0]['type']
    if qtype == 'choice':
        acc, chance = _mean([o['correct'] for o in outs]), _mean([o['chance'] for o in outs])
        return 100 * (acc - chance) / (1 - chance)
    if qtype == 'noul':
        return 100 * (_mean([o['correct'] for o in outs]) - 0.5) / 0.5
    chance = _mean([o['nmae_chance_mid'] for o in outs])
    if chance <= 0:  # every gold at mid-scale: the random-level baseline
        chance = _mean([o['nmae_chance'] for o in outs])
    return 100 * (1 - _mean([o['nmae'] for o in outs]) / chance)


def intelligence(outs: list[dict]) -> tuple[float | None, dict]:
    per_type = {}
    for qtype in TYPES:
        tiers = {t: competence([o for o in outs if o['type'] == qtype and o['tier'] == t]) for t in TIER_WEIGHTS}
        cc = _weighted(tiers, TIER_WEIGHTS)
        per_type[qtype] = None if cc is None else max(0.0, cc)
    total = _weighted(per_type, {t: 1 / 3 for t in TYPES})
    return (None if total is None else min(100.0, max(0.0, total))), per_type


def _ece_score(e: float | None) -> float | None:
    return None if e is None else max(0.0, 100 * (1 - e / 0.5))


def calibration(outs: list[dict]) -> tuple[float | None, dict]:
    parts = {}
    choice = [o for o in outs if o['type'] == 'choice' and o['valid']]
    if choice:
        parts['choice'] = _ece_score(ece([(o['conf'], float(o['correct'])) for o in choice if o['tier'] == 'hard']))
    noul = [o for o in outs if o['type'] == 'noul' and o['valid']]
    if noul:
        parts['noul'] = _ece_score(ece([(o['p_yes'], o['y']) for o in noul]))
    score = [o for o in outs if o['type'] == 'score' and o['valid']]
    if score:
        top = _ece_score(ece([(o['conf'], float(o['correct'])) for o in score]))
        parts['score'] = (top + 100 * (1 - _mean([o['nrps'] for o in score]))) / 2
    return _weighted(parts, {t: 1 / 3 for t in TYPES}), parts


def summary(outs: list[dict]) -> dict:
    i, cc = intelligence(outs)
    c, cal = calibration(outs)
    accuracy = {
        t: _mean([o['correct'] for o in outs if o['type'] == t]) * 100
        for t in TYPES
        if any(o['type'] == t for o in outs)
    }
    return {
        'capability': None if i is None or c is None else (i + c) / 2,
        'intelligence': i,
        'calibration': c,
        'competence_per_type': cc,
        'calibration_per_type': cal,
        'accuracy_per_type': accuracy,
        'items': len(outs),
        'predicted': sum(o['valid'] for o in outs),
    }


# ------------------------------------------------------------------ input


def normalized(task: dict, probs: dict) -> dict:
    labels = task['labels']
    if task['question']['type'] == 'noul' and 'no' not in probs:
        probs = {'yes': probs['yes'], 'no': 1.0 - probs['yes']}
    values = {lab: float(probs.get(lab, 0.0)) for lab in labels}
    if any(not math.isfinite(v) or v < 0 for v in values.values()):
        raise ValueError(f'{task["id"]}: probabilities must be finite and non-negative')
    if abs(sum(values.values()) - 1.0) > 0.01:
        raise ValueError(
            f'{task["id"]}: probabilities over the labels {labels} sum to {sum(values.values()):.4f}, not 1'
        )
    return values


def score(tasks: list[dict], predictions: list[dict]) -> dict:
    by_id = {t['id']: t for t in tasks}
    probs = {}
    for p in predictions:
        if p['id'] not in by_id:
            raise ValueError(f'unknown item id {p["id"]!r}')
        if p['id'] in probs:
            raise ValueError(f'item {p["id"]!r} predicted twice')
        probs[p['id']] = normalized(by_id[p['id']], p['probs'])
    return summary([item_outcome(t, probs.get(t['id'])) for t in tasks])


def aggregate_official(sample_scores: list, tasks: list) -> list:
    """Include every requested item in the official denominator, including missing predictions."""
    from evalscope.api.metric import AggScore

    predictions = []
    for sample in sample_scores:
        observation = (sample.score.metadata or {}).get('decision')
        metadata = sample.sample_metadata or {}
        if observation:
            predictions.append({'id': metadata['source_id'], 'probs': observation['raw_probabilities']})
    result = score(tasks, predictions)
    return [
        AggScore(
            metric_name=name,
            aggregation='official',
            score=value / 100,
            num=len(tasks),
            metadata={
                'upstream_metric': value,
                'predicted': result['predicted'],
                'missing_predictions_count_as_wrong': True,
                'unavailable': {
                    key: 'No valid forecasts for this official metric.'
                    for key in ['intelligence', 'calibration', 'capability']
                    if result[key] is None
                },
            },
        )
        for name in ['intelligence', 'calibration', 'capability']
        if (value := result[name]) is not None
    ]
