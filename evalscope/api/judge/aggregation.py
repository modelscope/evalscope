"""Pure aggregation helpers for Native judge reviews.

The executor owns request execution and review lifecycle. This module owns only deterministic
reduction of already parsed verdicts, so aggregation rules can be inspected without following
transport or adapter code.
"""
import math
import statistics
from collections import Counter, defaultdict
from pydantic import BaseModel
from typing import Any, Dict, Iterable, List, Optional, Sequence

from .types import JudgeObservation, PairwiseOutcome, PairwisePlacementOutcome


def aggregate_repeat_values(values: Sequence[Dict[str, float]], method: str) -> tuple[Dict[str, float], List[str]]:
    """Reduce one judge's repeats and return metrics that needed a first-observation tie-break."""
    buckets = _metric_buckets(values)
    if method == 'median':
        return {name: float(statistics.median(items)) for name, items in buckets.items()}, []
    if method == 'majority_vote':
        result: Dict[str, float] = {}
        ties: List[str] = []
        for name, items in buckets.items():
            winner, tied = _majority(items)
            if tied:
                # Repeats have no independent primary judge. The first valid observation is
                # deterministic; callers mark the review degraded and retain this diagnostic.
                winner = items[0]
                ties.append(name)
            result[name] = winner
        return result, ties
    return {name: sum(items) / len(items) for name, items in buckets.items()}, []


def aggregate_judge_values(judge_values: Dict[str, Dict[str, float]], method: str,
                           primary: str) -> tuple[Optional[Dict[str, float]], bool, List[str]]:
    """Combine equally weighted judges, using the primary judge only for a tied majority vote."""
    buckets = _metric_buckets(judge_values.values())
    result: Dict[str, float] = {}
    tie_broken = False
    unresolved_ties: List[str] = []
    for name, values in buckets.items():
        if method == 'median':
            result[name] = float(statistics.median(values))
        elif method == 'majority_vote':
            winner, tied = _majority(values)
            if tied:
                if primary not in judge_values or name not in judge_values[primary]:
                    unresolved_ties.append(name)
                    continue
                winner = judge_values[primary][name]
                tie_broken = True
            result[name] = winner
        else:
            result[name] = sum(values) / len(values)
    return result or None, tie_broken, unresolved_ties


def aggregate_pairwise_outcomes(observations: Sequence[JudgeObservation], primary: str,
                                min_valid_judges: int) -> tuple[Optional[PairwiseOutcome], bool]:
    """Vote candidate-oriented pairwise results and their presentation-order diagnostics."""
    per_judge: Dict[str, List[PairwiseOutcome]] = defaultdict(list)
    for observation in observations:
        outcome = observation.reduced.outcome if observation.reduced is not None else None
        if outcome is not None:
            per_judge[observation.judge_id].append(outcome)
    if len(per_judge) < min_valid_judges:
        return None, False

    judge_outcomes = {judge_id: _vote_pairwise(values, repeat_vote=True)[0] for judge_id, values in per_judge.items()}
    outcome, tied = _vote_pairwise(
        list(judge_outcomes.values()), repeat_vote=False, primary=judge_outcomes.get(primary)
    )
    if tied and primary not in judge_outcomes:
        return None, False

    placements: Dict[str, PairwisePlacementOutcome] = {}
    by_placement: Dict[str, Dict[str, List[PairwisePlacementOutcome]]] = defaultdict(lambda: defaultdict(list))
    for judge_id, outcomes in per_judge.items():
        for candidate_outcome in outcomes:
            for placement, placement_outcome in candidate_outcome.placements.items():
                by_placement[placement][judge_id].append(placement_outcome)
    for placement, per_judge_placements in by_placement.items():
        if len(per_judge_placements) < min_valid_judges:
            continue
        judge_placements = {
            judge_id: _vote_pairwise_placements(values, repeat_vote=True)[0]
            for judge_id, values in per_judge_placements.items()
        }
        placement_outcome, placement_tied = _vote_pairwise_placements(
            list(judge_placements.values()),
            repeat_vote=False,
            primary=judge_placements.get(primary),
        )
        if placement_tied and primary not in judge_placements:
            return None, False
        placements[placement] = placement_outcome
        tied = tied or placement_tied
    return outcome.model_copy(update={'placements': placements}), tied


def judge_disagreement(per_judge: Dict[str, List[Dict[str, float]]],
                       observations: Sequence[JudgeObservation]) -> Dict[str, Any]:
    """Build per-sample numeric, categorical, and position-consistency diagnostics."""
    all_values = _metric_buckets(value for values in per_judge.values() for value in values)
    numeric = {
        name: {
            'std': statistics.pstdev(values) if len(values) > 1 else 0.0,
            'range': max(values) - min(values),
        }
        for name, values in all_values.items()
    }
    repeat_numeric = {
        judge_id: _numeric_disagreement(values)
        for judge_id, values in per_judge.items()
        if len(values) > 1
    }
    per_judge_mean = {judge_id: aggregate_repeat_values(values, 'mean')[0] for judge_id, values in per_judge.items()}
    categorical = _categorical_disagreement_by_case(observations)
    positions = _position_consistency(observations)
    return {
        'numeric': {
            'all_observations': numeric,
            'repeats': repeat_numeric,
            'cross_judge': _numeric_disagreement(list(per_judge_mean.values())),
        },
        'categorical': categorical,
        'position_consistency': sum(positions) / len(positions) if positions else None,
        'swap_flip_count': sum(not value for value in positions),
    }


def _metric_buckets(values: Iterable[Dict[str, float]]) -> Dict[str, List[float]]:
    buckets: Dict[str, List[float]] = defaultdict(list)
    for value in values:
        for name, item in value.items():
            buckets[name].append(float(item))
    return buckets


def _majority(values: Sequence[float]) -> tuple[float, bool]:
    counts = Counter(values)
    top = max(counts.values())
    winners = [value for value, count in counts.items() if count == top]
    return winners[0], len(winners) > 1


def _numeric_disagreement(values: Sequence[Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    return {
        name: {
            'std': statistics.pstdev(items) if len(items) > 1 else 0.0,
            'range': max(items) - min(items),
        }
        for name, items in _metric_buckets(values).items()
    }


def _categorical_disagreement_by_case(observations: Sequence[JudgeObservation]) -> Dict[str, Dict[str, Any]]:
    categories: Dict[str, List[tuple[str, str]]] = defaultdict(list)
    for observation in observations:
        outcome = observation.reduced.outcome if observation.reduced is not None else None
        if outcome is not None:
            categories[f'pairwise/{outcome.metric_name}'].append((observation.judge_id, outcome.result))
            continue
        for verdict in observation.case_verdicts:
            label = _categorical_label(verdict.value)
            if label is not None:
                categories[verdict.case_id].append((observation.judge_id, label))

    result: Dict[str, Dict[str, Any]] = {}
    for case_id, labels in categories.items():
        per_judge: Dict[str, List[str]] = defaultdict(list)
        for judge_id, label in labels:
            per_judge[judge_id].append(label)
        result[case_id] = {
            **_categorical_disagreement([label for _, label in labels]),
            'repeats': {
                judge_id: _categorical_disagreement(values)
                for judge_id, values in per_judge.items()
                if len(values) > 1
            },
            'cross_judge': _categorical_disagreement([_majority_label(values) for values in per_judge.values()]),
        }
    return result


def _categorical_label(value: Any) -> Optional[str]:
    if isinstance(value, BaseModel) and hasattr(value, 'verdict'):
        return str(value.verdict)
    return value if isinstance(value, str) else None


def _categorical_disagreement(labels: Sequence[str]) -> Dict[str, float]:
    if not labels:
        return {'agreement_ratio': 0.0, 'vote_entropy': 0.0}
    counts = Counter(labels)
    total = len(labels)
    entropy = -sum((count / total) * math.log2(count / total) for count in counts.values())
    return {'agreement_ratio': max(counts.values()) / total, 'vote_entropy': entropy}


def _majority_label(labels: Sequence[str]) -> str:
    return Counter(labels).most_common(1)[0][0]


def _position_consistency(observations: Sequence[JudgeObservation]) -> List[bool]:
    positions: List[bool] = []
    for observation in observations:
        reduced = observation.reduced
        outcome = reduced.outcome if reduced is not None else None
        if outcome is not None and outcome.placements:
            positions.append(len({value.result for value in outcome.placements.values()}) == 1)
        elif reduced is not None and reduced.position_results:
            positions.append(len(set(reduced.position_results.values())) == 1)
    return positions


def _vote_pairwise(outcomes: Sequence[PairwiseOutcome],
                   repeat_vote: bool,
                   primary: Optional[PairwiseOutcome] = None) -> tuple[PairwiseOutcome, bool]:
    """Return a semantic majority; ties within a judge's repeats deliberately become a draw."""
    counts = Counter(outcome.result for outcome in outcomes)
    top = max(counts.values())
    winners = [result for result, count in counts.items() if count == top]
    tied = len(winners) > 1
    result = 'tie' if tied and repeat_vote else primary.result if tied and primary is not None else winners[0]
    selected = [outcome for outcome in outcomes if outcome.result == result]
    strength = 'weak'
    if result != 'tie' and selected:
        strength = 'strong' if sum(item.strength == 'strong' for item in selected) * 2 > len(selected) else 'weak'
    template = selected[0] if selected else outcomes[0]
    return PairwiseOutcome(metric_name=template.metric_name, result=result, strength=strength), tied


def _vote_pairwise_placements(
    outcomes: Sequence[PairwisePlacementOutcome],
    repeat_vote: bool,
    primary: Optional[PairwisePlacementOutcome] = None,
) -> tuple[PairwisePlacementOutcome, bool]:
    proxies = [
        PairwiseOutcome(metric_name='placement', result=item.result, strength=item.strength) for item in outcomes
    ]
    primary_proxy = (
        PairwiseOutcome(metric_name='placement', result=primary.result, strength=primary.strength)
        if primary is not None else None
    )
    voted, tied = _vote_pairwise(proxies, repeat_vote, primary_proxy)
    return PairwisePlacementOutcome(result=voted.result, strength=voted.strength), tied
