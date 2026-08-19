"""The single execution path for Native LLM judge contracts."""
import math
import statistics
import time
from collections import Counter, defaultdict
from pydantic import BaseModel, ConfigDict, Field
from typing import Any, Dict, List, Literal, Optional, Sequence

from evalscope.api.metric import JudgeSummary, Score
from evalscope.constants import ScoreStatus
from evalscope.utils.logger import get_logger
from .types import (
    CaseVerdict,
    JudgeAttempt,
    JudgeCase,
    JudgeContext,
    JudgeObservation,
    JudgeProtocol,
    JudgeReview,
    Placement,
)

logger = get_logger()

MAX_STAGES = 4


class JudgeExecutorConfig(BaseModel):
    """The resolved execution policy for one benchmark run."""

    model_config = ConfigDict(frozen=True)

    repeats: int = Field(default=1, ge=1)
    position_swap: bool = False
    aggregation: Literal['mean', 'median', 'majority_vote'] = 'mean'
    min_valid_judges: int = Field(default=1, ge=1)


class JudgeExecutor:
    """Execute one complete judge session at a time, serially within each sample."""

    def __init__(self, judges: Sequence[Any], config: Optional[JudgeExecutorConfig] = None) -> None:
        self.judges = list(judges)
        self.config = config or JudgeExecutorConfig()
        if not self.judges:
            raise ValueError('JudgeExecutor requires at least one judge model.')

    def execute(self, adapter: JudgeProtocol, context: JudgeContext) -> JudgeReview:
        """Score one sample through every configured judge and repeat."""
        review = JudgeReview()
        for judge in self.judges:
            for repeat_id in range(self.config.repeats):
                review.observations.append(self._run_observation(adapter, context, judge, repeat_id, review))
        review.failure_counts = dict(
            Counter(attempt.status.value for attempt in review.attempts if not attempt.status.is_usable)
        )
        self._aggregate(review)
        return review

    def _run_observation(
        self,
        adapter: JudgeProtocol,
        context: JudgeContext,
        judge: Any,
        repeat_id: int,
        review: JudgeReview,
    ) -> JudgeObservation:
        observation = JudgeObservation(judge_id=_judge_id(judge), repeat_id=repeat_id)
        pending = list(adapter.build_judge_cases(context))
        declared = len(pending)
        completed: List[CaseVerdict] = []
        for stage in range(MAX_STAGES):
            for case in pending:
                verdict = self._run_case(adapter, context, judge, repeat_id, case, completed, review)
                if verdict is None:
                    observation.status = ScoreStatus.INVALID_SESSION
                    observation.error = f'case {case.case_id} produced no usable verdict'
                    return observation
                completed.append(verdict)
            pending = list(adapter.expand_judge_cases(stage + 1, completed, context))
            declared += len(pending)
        if pending:
            raise RuntimeError(f'Judge case expansion exceeded {MAX_STAGES} stages; adapter bug.')
        if declared and not completed:
            observation.status = ScoreStatus.EXCLUDED
            observation.error = 'no case produced a usable verdict'
            return observation
        observation.case_verdicts = completed
        observation.reduced = adapter.reduce_judge_verdicts(completed, context)
        if any(verdict.status is ScoreStatus.FALLBACK for verdict in completed):
            observation.status = ScoreStatus.FALLBACK
        return observation

    def _run_case(
        self,
        adapter: JudgeProtocol,
        context: JudgeContext,
        judge: Any,
        repeat_id: int,
        case: JudgeCase,
        completed: Sequence[CaseVerdict],
        review: JudgeReview,
    ) -> Optional[CaseVerdict]:
        placements = (Placement.ORIGINAL, Placement.SWAPPED) if self.config.position_swap else (Placement.ORIGINAL, )
        values: Dict[str, Any] = {}
        for placement in placements:
            value = self._resolve_placement(adapter, context, judge, repeat_id, case, placement, completed, review)
            if value is None:
                fallback = adapter.judge_fallback_verdict(case, context)
                if fallback is not None:
                    return fallback.model_copy(
                        update={
                            'status': ScoreStatus.FALLBACK,
                            'metadata': fallback.metadata or dict(case.metadata)
                        }
                    )
                return None
            values[placement.value] = value
        if len(placements) == 1:
            return CaseVerdict(
                case_id=case.case_id, value=values[Placement.ORIGINAL.value], metadata=dict(case.metadata)
            )
        return CaseVerdict(
            case_id=case.case_id,
            value=[values[Placement.ORIGINAL.value], values[Placement.SWAPPED.value]],
            placements=values,
            metadata=dict(case.metadata),
        )

    def _resolve_placement(
        self,
        adapter: JudgeProtocol,
        context: JudgeContext,
        judge: Any,
        repeat_id: int,
        case: JudgeCase,
        placement: Placement,
        completed: Sequence[CaseVerdict],
        review: JudgeReview,
    ) -> Any:
        """Make exactly one request. Transport retry belongs to the model implementation."""
        request = adapter.build_judge_request(case, placement, completed, context)
        started = time.perf_counter()
        try:
            output = judge.generate(list(request.messages))
        except Exception as exc:  # provider exceptions become inspectable failed attempts
            review.attempts.append(
                JudgeAttempt(
                    status=ScoreStatus.TRANSPORT_ERROR,
                    case_id=case.case_id,
                    judge_id=_judge_id(judge),
                    repeat_id=repeat_id,
                    placement=placement,
                    messages=list(request.messages),
                    error=f'{type(exc).__name__}: {exc}',
                    latency=time.perf_counter() - started,
                )
            )
            return None
        raw = output.completion
        attempt = JudgeAttempt(
            status=ScoreStatus.SUCCESS,
            case_id=case.case_id,
            judge_id=_judge_id(judge),
            repeat_id=repeat_id,
            placement=placement,
            messages=list(request.messages),
            model_output=output,
            raw_response=raw,
            latency=time.perf_counter() - started,
        )
        result = case.output_contract.parse(raw)
        if result.ok:
            attempt.parsed_value = _jsonable(result.value)
            review.attempts.append(attempt)
            return result.value
        attempt.status = ScoreStatus.PARSE_ERROR
        attempt.error = result.error
        review.attempts.append(attempt)
        logger.warning(f'Judge {_judge_id(judge)} failed the output contract for case {case.case_id}; excluding it.')
        return None

    def _aggregate(self, review: JudgeReview) -> None:
        """Reduce repeats inside a judge, then combine equally weighted judge outcomes."""
        per_judge: Dict[str, List[Dict[str, float]]] = defaultdict(list)
        for observation in review.valid_observations:
            per_judge[observation.judge_id].append(observation.reduced.value)
        if not per_judge:
            review.status = ScoreStatus.EXCLUDED
            review.error = 'no judge produced a usable verdict'
            return
        if len(per_judge) < self.config.min_valid_judges:
            review.status = ScoreStatus.EXCLUDED
            review.error = f'only {len(per_judge)} judge(s) produced valid verdicts; need {self.config.min_valid_judges}'
            return
        judge_values = {
            judge_id: _aggregate_values(values, self.config.aggregation)
            for judge_id, values in per_judge.items()
        }
        combined, tie_broken = _aggregate_across_judges(
            judge_values, self.config.aggregation, _judge_id(self.judges[0])
        )
        if combined is None:
            review.status = ScoreStatus.EXCLUDED
            review.error = 'primary judge has no valid verdict to break a majority_vote tie'
            return
        review.value = combined
        review.outcome = _aggregate_outcomes(review.valid_observations, _judge_id(self.judges[0]))
        review.disagreement = _disagreement(per_judge, review.valid_observations)
        expected = len(self.judges) * self.config.repeats
        if tie_broken:
            review.metadata['tie_broken_by_primary'] = True
            review.status = ScoreStatus.DEGRADED
        elif len(review.valid_observations) != expected or any(
            observation.status is ScoreStatus.FALLBACK for observation in review.valid_observations
        ):
            review.status = ScoreStatus.DEGRADED

    def build_score(self, adapter: JudgeProtocol, review: JudgeReview, context: JudgeContext) -> Score:
        """Build the final score and persist all judge I/O in the review payload."""
        score = adapter.finalize_judge_score(review, context)
        score.status = review.status
        if not review.status.is_usable:
            score.value = {}
        score.judge_summary = JudgeSummary(
            status=review.status,
            scored=int(review.status.is_usable),
            total=1,
            coverage=float(review.status.is_usable),
            judge_models=[_judge_id(judge) for judge in self.judges],
            valid_observations=len(review.valid_observations),
            total_observations=len(review.observations),
            failures=review.failure_counts,
            disagreement=review.disagreement,
            error=review.error,
        )
        score.metadata = dict(score.metadata or {})
        score.metadata['judge_attempts'] = [attempt.model_dump(exclude_none=True) for attempt in review.attempts]
        return score


def _judge_id(judge: Any) -> str:
    return str(getattr(judge, 'judge_id', getattr(judge, 'model_id', 'unknown-judge')))


def _aggregate_values(values: Sequence[Dict[str, float]], method: str) -> Dict[str, float]:
    buckets: Dict[str, List[float]] = defaultdict(list)
    for value in values:
        for name, item in value.items():
            buckets[name].append(float(item))
    if method == 'median':
        return {name: float(statistics.median(items)) for name, items in buckets.items()}
    if method == 'majority_vote':
        return {name: _majority(items)[0] for name, items in buckets.items()}
    return {name: sum(items) / len(items) for name, items in buckets.items()}


def _aggregate_across_judges(judge_values: Dict[str, Dict[str, float]], method: str,
                             primary: str) -> tuple[Optional[Dict[str, float]], bool]:
    buckets: Dict[str, List[float]] = defaultdict(list)
    for values in judge_values.values():
        for name, value in values.items():
            buckets[name].append(value)
    result: Dict[str, float] = {}
    tie_broken = False
    for name, values in buckets.items():
        if method == 'median':
            result[name] = float(statistics.median(values))
        elif method == 'majority_vote':
            winner, tied = _majority(values)
            if tied:
                if primary not in judge_values or name not in judge_values[primary]:
                    return None, False
                winner = judge_values[primary][name]
                tie_broken = True
            result[name] = winner
        else:
            result[name] = sum(values) / len(values)
    return result, tie_broken


def _majority(values: Sequence[float]) -> tuple[float, bool]:
    counts = Counter(values)
    top = max(counts.values())
    winners = [value for value, count in counts.items() if count == top]
    return winners[0], len(winners) > 1


def _disagreement(per_judge: Dict[str, List[Dict[str, float]]],
                  observations: Sequence[JudgeObservation]) -> Dict[str, Any]:
    all_values: Dict[str, List[float]] = defaultdict(list)
    for judge_id, values in per_judge.items():
        for value in values:
            for name, item in value.items():
                all_values[name].append(float(item))
    numeric = {
        name: {
            'std': statistics.pstdev(values) if len(values) > 1 else 0.0,
            'range': max(values) - min(values)
        }
        for name, values in all_values.items()
    }
    repeat_numeric = {
        judge_id: _numeric_disagreement(values)
        for judge_id, values in per_judge.items()
        if len(values) > 1
    }
    per_judge_aggregate = {judge_id: _aggregate_values(values, 'mean') for judge_id, values in per_judge.items()}
    cross_judge_numeric = _numeric_disagreement(list(per_judge_aggregate.values()))
    categorical: Dict[str, Dict[str, Any]] = {}
    categories: Dict[str, List[tuple[str, str]]] = defaultdict(list)
    for observation in observations:
        for verdict in observation.case_verdicts:
            label = _categorical_label(verdict.value)
            if label is not None:
                categories[verdict.case_id].append((observation.judge_id, label))
    for case_id, labels in categories.items():
        all_labels = [label for _, label in labels]
        per_judge_labels: Dict[str, List[str]] = defaultdict(list)
        for judge_id, label in labels:
            per_judge_labels[judge_id].append(label)
        per_judge_vote = [_majority_label(values) for values in per_judge_labels.values()]
        categorical[case_id] = {
            **_categorical_disagreement(all_labels),
            'repeats': {
                judge_id: _categorical_disagreement(values)
                for judge_id, values in per_judge_labels.items()
                if len(values) > 1
            },
            'cross_judge': _categorical_disagreement(per_judge_vote),
        }
    positions = []
    for observation in observations:
        for verdict in observation.case_verdicts:
            if verdict.placements:
                positions.append(len({str(value) for value in verdict.placements.values()}) == 1)
    return {
        'numeric': {
            'all_observations': numeric,
            'repeats': repeat_numeric,
            'cross_judge': cross_judge_numeric,
        },
        'categorical': categorical,
        'position_consistency': sum(positions) / len(positions) if positions else None,
        'swap_flip_count': sum(not value for value in positions),
    }


def _numeric_disagreement(values: Sequence[Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    buckets: Dict[str, List[float]] = defaultdict(list)
    for value in values:
        for name, item in value.items():
            buckets[name].append(float(item))
    return {
        name: {
            'std': statistics.pstdev(items) if len(items) > 1 else 0.0,
            'range': max(items) - min(items),
        }
        for name, items in buckets.items()
    }


def _categorical_label(value: Any) -> Optional[str]:
    if isinstance(value, BaseModel) and hasattr(value, 'verdict'):
        return str(value.verdict)
    if isinstance(value, str):
        return value
    return None


def _categorical_disagreement(labels: Sequence[str]) -> Dict[str, float]:
    if not labels:
        return {'agreement_ratio': 0.0, 'vote_entropy': 0.0}
    counts = Counter(labels)
    total = len(labels)
    entropy = -sum((count / total) * math.log2(count / total) for count in counts.values())
    return {
        'agreement_ratio': max(counts.values()) / total,
        'vote_entropy': entropy,
    }


def _majority_label(labels: Sequence[str]) -> str:
    return Counter(labels).most_common(1)[0][0]


def _aggregate_outcomes(observations: Sequence[JudgeObservation], primary: str) -> Any:
    """Aggregate typed categorical outcomes by repeat, then by judge.

    Only mappings with a common scalar leaf structure are aggregated. Other adapter-specific
    diagnostic payloads remain observation metadata and never influence final scoring.
    """
    per_judge: Dict[str, List[Any]] = defaultdict(list)
    for observation in observations:
        if observation.reduced is not None and observation.reduced.outcome is not None:
            per_judge[observation.judge_id].append(observation.reduced.outcome)
    if not per_judge:
        return None

    per_judge_outcome = {judge_id: _majority_outcome(values, primary=False) for judge_id, values in per_judge.items()}
    return _majority_outcome(
        list(per_judge_outcome.values()),
        primary=True,
        primary_value=per_judge_outcome.get(primary),
    )


def _majority_outcome(values: Sequence[Any], primary: bool, primary_value: Any = None) -> Any:
    """Vote recursively over mapping leaves; only identical shapes are meaningful outcomes."""
    if not values:
        return None
    if all(isinstance(value, dict) for value in values):
        keys = set(values[0])
        if any(set(value) != keys for value in values):
            return None
        return {
            key: _majority_outcome(
                [value[key]
                 for value in values],
                primary,
                primary_value.get(key) if isinstance(primary_value, dict) else None,
            )
            for key in keys
        }
    counts = Counter(_freeze_outcome(value) for value in values)
    winner_count = max(counts.values())
    winners = [value for value in values if counts[_freeze_outcome(value)] == winner_count]
    if len({_freeze_outcome(value) for value in winners}) == 1:
        return winners[0]
    return primary_value if primary and primary_value is not None else winners[0]


def _freeze_outcome(value: Any) -> str:
    return repr(value)


def _jsonable(value: Any) -> Any:
    return value.model_dump() if isinstance(value, BaseModel) else value


__all__: Sequence[str] = ('JudgeExecutor', 'JudgeExecutorConfig')
