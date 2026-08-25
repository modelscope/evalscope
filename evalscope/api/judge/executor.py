"""The single execution path for Native LLM judge contracts."""
import time
from collections import Counter, defaultdict
from typing import Any, Dict, List, Literal, Optional, Sequence

from pydantic import BaseModel, ConfigDict, Field

from evalscope.api.metric import JudgeSummary, Score
from evalscope.constants import ScoreStatus
from evalscope.utils.logger import get_logger

from .aggregation import (
    aggregate_judge_values,
    aggregate_pairwise_outcomes,
    aggregate_repeat_values,
    judge_disagreement,
)
from .definition import JudgeDefinition
from .types import CaseVerdict, JudgeAttempt, JudgeCase, JudgeContext, JudgeObservation, JudgeReview, Placement

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
        if self.config.min_valid_judges > len(self.judges):
            raise ValueError('min_valid_judges cannot exceed the configured judge count.')

    def execute(self, definition: JudgeDefinition, context: JudgeContext) -> JudgeReview:
        """Score one sample through every configured judge and repeat."""
        review = JudgeReview()
        for judge in self.judges:
            for repeat_id in range(self.config.repeats):
                review.observations.append(self._run_observation(definition, context, judge, repeat_id, review))
        review.failure_counts = dict(
            Counter(attempt.status.value for attempt in review.attempts if not attempt.status.is_usable)
        )
        self._aggregate(review)
        return review

    def _run_observation(
        self,
        definition: JudgeDefinition,
        context: JudgeContext,
        judge: Any,
        repeat_id: int,
        review: JudgeReview,
    ) -> JudgeObservation:
        observation = JudgeObservation(judge_id=_judge_id(judge), repeat_id=repeat_id)
        pending = list(definition.build_cases(context))
        declared = len(pending)
        completed: List[CaseVerdict] = []
        for stage in range(MAX_STAGES):
            for case in pending:
                verdict = self._run_case(definition, context, judge, repeat_id, case, completed, review)
                if verdict is None:
                    observation.status = ScoreStatus.INVALID_SESSION
                    observation.error = f'case {case.case_id} produced no usable verdict'
                    return observation
                completed.append(verdict)
            pending = list(definition.expand_cases(stage + 1, completed, context))
            declared += len(pending)
        if pending:
            raise RuntimeError(f'Judge case expansion exceeded {MAX_STAGES} stages; adapter bug.')
        if declared and not completed:
            observation.status = ScoreStatus.EXCLUDED
            observation.error = 'no case produced a usable verdict'
            return observation
        observation.case_verdicts = completed
        observation.reduced = definition.reduce_verdicts(completed, context)
        if any(verdict.status is ScoreStatus.FALLBACK for verdict in completed):
            observation.status = ScoreStatus.FALLBACK
        return observation

    def _run_case(
        self,
        definition: JudgeDefinition,
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
            value = self._resolve_placement(definition, context, judge, repeat_id, case, placement, completed, review)
            if value is None:
                # A swapped pair is atomic. One rule fallback must not impersonate the missing
                # opposite presentation.
                if len(placements) > 1:
                    return None
                fallback = definition.fallback_verdict(case, context)
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
        definition: JudgeDefinition,
        context: JudgeContext,
        judge: Any,
        repeat_id: int,
        case: JudgeCase,
        placement: Placement,
        completed: Sequence[CaseVerdict],
        review: JudgeReview,
    ) -> Any:
        """Make exactly one request. Transport retry belongs to the model implementation."""
        request = definition.build_request(case, placement, completed, context)
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
        pairwise_metrics: Dict[str, set[str]] = defaultdict(set)
        for observation in review.valid_observations:
            per_judge[observation.judge_id].append(observation.reduced.value)
            if observation.reduced.outcome is not None:
                pairwise_metrics[observation.judge_id].add(observation.reduced.outcome.metric_name)
        if not per_judge:
            if review.fallback_observations:
                self._aggregate_rule_fallback(review)
                return
            review.status = ScoreStatus.EXCLUDED
            review.error = 'no judge produced a usable verdict'
            return
        if len(per_judge) < self.config.min_valid_judges:
            if review.fallback_observations:
                self._aggregate_rule_fallback(review)
                return
            review.status = ScoreStatus.EXCLUDED
            review.error = f'only {len(per_judge)} judge(s) produced valid verdicts; need {self.config.min_valid_judges}'
            return
        judge_values: Dict[str, Dict[str, float]] = {}
        repeat_ties: Dict[str, List[str]] = {}
        for judge_id, values in per_judge.items():
            judge_values[judge_id], ties = aggregate_repeat_values(values, self.config.aggregation)
            ties = [metric_name for metric_name in ties if metric_name not in pairwise_metrics[judge_id]]
            if ties:
                repeat_ties[judge_id] = ties
        eligible_metrics = {
            metric_name
            for values in judge_values.values()
            for metric_name in values
            if sum(metric_name in judge_value for judge_value in judge_values.values()) >= self.config.min_valid_judges
        }
        if not eligible_metrics:
            review.status = ScoreStatus.EXCLUDED
            review.error = 'no metric received enough valid judge verdicts'
            return
        excluded_metrics = sorted({metric_name
                                   for values in judge_values.values()
                                   for metric_name in values} - eligible_metrics)
        judge_values = {
            judge_id: {
                metric_name: value
                for metric_name, value in values.items()
                if metric_name in eligible_metrics
            }
            for judge_id, values in judge_values.items()
        }
        combined, tie_broken, unresolved_ties = aggregate_judge_values(
            judge_values, self.config.aggregation, _judge_id(self.judges[0])
        )
        if combined is None:
            review.status = ScoreStatus.EXCLUDED
            review.error = 'no metric could be aggregated after applying the majority_vote tie-break rule'
            return
        review.value = combined
        pairwise, pairwise_tie_broken = aggregate_pairwise_outcomes(
            review.valid_observations, _judge_id(self.judges[0]), self.config.min_valid_judges
        )
        if pairwise is not None:
            review.outcome = pairwise
            review.value[pairwise.metric_name] = pairwise.score
        tie_broken = tie_broken or pairwise_tie_broken
        primary_observation = next(
            (
                observation for observation in review.valid_observations
                if observation.judge_id == _judge_id(self.judges[0])
            ),
            review.valid_observations[0],
        )
        review.metadata = dict(primary_observation.reduced.metadata)
        review.observation_metadata = [{
            'judge_id': observation.judge_id,
            'repeat_id': observation.repeat_id,
            'status': observation.status.value,
            'metadata': dict(observation.reduced.metadata),
        } for observation in review.usable_observations]
        review.disagreement = judge_disagreement(per_judge, review.valid_observations)
        if excluded_metrics:
            review.metadata['metrics_without_quorum'] = excluded_metrics
        if unresolved_ties:
            review.metadata['metrics_without_primary_tiebreak'] = unresolved_ties
        if repeat_ties:
            review.metadata['repeat_tie_broken_by_first_observation'] = repeat_ties
        expected = len(self.judges) * self.config.repeats
        if tie_broken:
            review.metadata['tie_broken_by_primary'] = True
            review.status = ScoreStatus.DEGRADED
        elif len(review.valid_observations) != expected or unresolved_ties or repeat_ties:
            review.status = ScoreStatus.DEGRADED

    def _aggregate_rule_fallback(self, review: JudgeReview) -> None:
        """Use one deterministic official fallback without treating it as a Judge vote."""
        primary = _judge_id(self.judges[0])
        fallback = next(
            (observation for observation in review.fallback_observations if observation.judge_id == primary),
            review.fallback_observations[0],
        )
        review.value = dict(fallback.reduced.value)
        review.outcome = fallback.reduced.outcome
        review.metadata = dict(fallback.reduced.metadata)
        review.observation_metadata = [{
            'judge_id': observation.judge_id,
            'repeat_id': observation.repeat_id,
            'status': observation.status.value,
            'metadata': dict(observation.reduced.metadata),
        } for observation in review.fallback_observations]
        review.status = ScoreStatus.DEGRADED

    def build_score(
        self,
        definition: JudgeDefinition,
        review: JudgeReview,
        context: JudgeContext,
        *,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Score:
        """Build the final score and persist all judge I/O in the review payload."""
        score = Score(
            extracted_prediction=context.filtered_prediction,
            prediction=context.original_prediction,
            value=dict(review.value),
            main_score_name=definition.main_score_name,
            metadata=dict(metadata or {}),
        )
        if definition.finalize is not None:
            score = definition.finalize(score, review, context)
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
        score.metadata['judge_observation_metadata'] = review.observation_metadata
        return score


def _judge_id(judge: Any) -> str:
    return str(getattr(judge, 'judge_id', None) or getattr(judge, 'model_id', 'unknown-judge'))


def _jsonable(value: Any) -> Any:
    return value.model_dump() if isinstance(value, BaseModel) else value


__all__: Sequence[str] = ('JudgeExecutor', 'JudgeExecutorConfig')
