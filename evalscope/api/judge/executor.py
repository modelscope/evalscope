"""Drives every judge call for one sample: repeats, stages, cases, placements, aggregation.

Retry boundaries are deliberately separate:

- transport failures are retried by the model layer (``GenerateConfig.retries``); the executor
  does not add its own network retry;
- a response that arrived but does not satisfy the output contract is retried only as many times
  as the contract declares, and each try is recorded as its own ``JudgeAttempt``;
- judge repeats are planned observations, never failure compensation.
"""
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pydantic import BaseModel, ConfigDict, Field
from typing import Any, Dict, List, Optional, Sequence

from evalscope.api.metric import JudgeDetail, Score
from evalscope.constants import ScoreStatus
from evalscope.metrics.judge.llm_judge import JUDGE_ERROR_PREFIX, LLMJudge
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

MAX_STAGES = 8
"""Stage expansion is bounded so a buggy adapter cannot loop up unbounded judge cost."""


class JudgeExecutorConfig(BaseModel):
    """Runtime knobs for one executor. Multi-judge and repeats are reserved, not yet enabled."""

    model_config = ConfigDict(frozen=True)

    repeats: int = Field(default=1, ge=1)
    position_swap: bool = False
    save_io: bool = True


class JudgeExecutor:
    """Executes a :class:`JudgeProtocol` adapter's cases against one or more judge models."""

    def __init__(
        self,
        judges: Sequence[LLMJudge],
        config: Optional[JudgeExecutorConfig] = None,
    ) -> None:
        self.judges = list(judges)
        self.config = config or JudgeExecutorConfig()
        if not self.judges:
            raise ValueError('JudgeExecutor requires at least one judge model.')
        if len(self.judges) > 1:
            raise ValueError('Multiple judge models are not supported yet; pass exactly one.')
        if self.config.repeats != 1:
            raise ValueError('Judge repeats are not supported yet; use repeats=1.')

    # ------------------------------------------------------------------
    # Entry points
    # ------------------------------------------------------------------

    def execute(self, adapter: JudgeProtocol, context: JudgeContext) -> JudgeReview:
        """Score one sample."""
        review = JudgeReview()
        for judge in self.judges:
            for repeat_id in range(self.config.repeats):
                observation = self._run_observation(adapter, context, judge, repeat_id, review)
                review.observations.append(observation)

        review.failure_counts = dict(
            Counter(attempt.status.value for attempt in review.attempts if not attempt.status.is_usable)
        )
        self._aggregate(adapter, review, context)
        return review

    def execute_batch(
        self,
        adapter: JudgeProtocol,
        contexts: Sequence[JudgeContext],
        max_workers: int = 1,
    ) -> List[JudgeReview]:
        """Score many samples, concurrently across samples but serially within one sample.

        Batch-scoring benchmarks route through here instead of building their own thread pool, so
        judge concurrency, failure counting and cost live in one place.
        """
        if not contexts:
            return []
        workers = max(1, min(int(max_workers), len(contexts)))
        if workers == 1:
            return [self.execute(adapter, context) for context in contexts]
        with ThreadPoolExecutor(max_workers=workers) as pool:
            return list(pool.map(lambda context: self.execute(adapter, context), contexts))

    # ------------------------------------------------------------------
    # One observation
    # ------------------------------------------------------------------

    def _run_observation(
        self,
        adapter: JudgeProtocol,
        context: JudgeContext,
        judge: LLMJudge,
        repeat_id: int,
        review: JudgeReview,
    ) -> JudgeObservation:
        judge_id = judge.model_id
        observation = JudgeObservation(judge_id=judge_id, repeat_id=repeat_id)

        pending = list(adapter.build_judge_cases(context))
        declared = len(pending)
        completed: List[CaseVerdict] = []
        for stage in range(MAX_STAGES):
            if not pending:
                break
            for case in pending:
                verdict = self._run_case(adapter, context, judge, repeat_id, case, completed, review)
                if verdict is None:
                    if case.required:
                        observation.status = ScoreStatus.INVALID_SESSION
                        observation.error = f'required case {case.case_id} produced no usable verdict'
                        return observation
                    continue
                completed.append(verdict)
            pending = list(adapter.expand_judge_cases(stage + 1, completed, context))
        if pending:
            raise RuntimeError(f'Judge case expansion exceeded {MAX_STAGES} stages; adapter bug.')

        observation.case_verdicts = completed
        if declared and not completed:
            # Cases were asked for but none survived; the sample has no usable judge verdict.
            observation.status = ScoreStatus.EXCLUDED
            observation.error = 'no case produced a usable verdict'
            return observation

        # Declaring no cases is legitimate: the adapter's rules already settled the sample.
        observation.reduced = adapter.reduce_judge_verdicts(completed, context)
        if any(verdict.status is ScoreStatus.FALLBACK for verdict in completed):
            observation.status = ScoreStatus.FALLBACK
        return observation

    def _run_case(
        self,
        adapter: JudgeProtocol,
        context: JudgeContext,
        judge: LLMJudge,
        repeat_id: int,
        case: JudgeCase,
        completed: Sequence[CaseVerdict],
        review: JudgeReview,
    ) -> Optional[CaseVerdict]:
        """Resolve one case over all required placements, or fall back, or give up."""
        placements = (Placement.ORIGINAL, Placement.SWAPPED) if self.config.position_swap else (Placement.ORIGINAL, )

        values: Dict[str, Any] = {}
        for placement in placements:
            value = self._resolve_placement(adapter, context, judge, repeat_id, case, placement, completed, review)
            if value is None:
                # Both sides of a swap form one atomic observation: a single successful side is
                # not half a verdict.
                fallback = adapter.judge_fallback_verdict(case, context)
                if fallback is not None:
                    return fallback.model_copy(update={'status': ScoreStatus.FALLBACK})
                return None
            values[placement.value] = value

        if len(placements) == 1:
            return CaseVerdict(case_id=case.case_id, value=values[Placement.ORIGINAL.value])
        return CaseVerdict(
            case_id=case.case_id,
            value=[values[Placement.ORIGINAL.value], values[Placement.SWAPPED.value]],
            placements=values,
        )

    def _resolve_placement(
        self,
        adapter: JudgeProtocol,
        context: JudgeContext,
        judge: LLMJudge,
        repeat_id: int,
        case: JudgeCase,
        placement: Placement,
        completed: Sequence[CaseVerdict],
        review: JudgeReview,
    ) -> Any:
        """Send one request, parse it strictly, and retry only as the contract allows."""
        contract = case.output_contract
        request = adapter.build_judge_request(case, placement, completed, context)

        for attempt_index in range(contract.parse_retries + 1):
            started = time.perf_counter()
            raw = judge.judge(messages=request.messages)
            latency = time.perf_counter() - started

            attempt = JudgeAttempt(
                status=ScoreStatus.SUCCESS,
                case_id=case.case_id,
                judge_id=judge.model_id,
                repeat_id=repeat_id,
                placement=placement,
                attempt_index=attempt_index,
                raw_response=raw if self.config.save_io else None,
                latency=latency,
            )

            if raw is None or raw.startswith(JUDGE_ERROR_PREFIX):
                # ``judge`` reports a failed request as an [ERROR] string; its embedded digits and
                # letters must never reach a parser.
                attempt.status = ScoreStatus.TRANSPORT_ERROR
                attempt.error = raw
                review.attempts.append(attempt)
                return None

            result = contract.parse(raw)
            if result.ok:
                attempt.parsed_value = _jsonable(result.value)
                review.attempts.append(attempt)
                return result.value

            attempt.status = ScoreStatus.PARSE_ERROR
            attempt.error = result.error
            review.attempts.append(attempt)

        logger.warning(
            f'Judge {judge.model_id} failed the output contract for case {case.case_id} after '
            f'{contract.parse_retries + 1} attempt(s); the sample is excluded from this metric.'
        )
        return None

    # ------------------------------------------------------------------
    # Aggregation
    # ------------------------------------------------------------------

    def _aggregate(self, adapter: JudgeProtocol, review: JudgeReview, context: JudgeContext) -> None:
        """Aggregate repeats within a judge, then judges with equal weight."""
        per_judge: Dict[str, List[Dict[str, float]]] = {}
        for observation in review.valid_observations:
            per_judge.setdefault(observation.judge_id, []).append(observation.reduced.value)

        if not per_judge:
            review.status = ScoreStatus.EXCLUDED
            review.value = {}
            review.error = review.error or 'no judge produced a usable verdict'
            return

        judge_values = [_combine(values) for values in per_judge.values()]
        review.value = _combine(judge_values)
        # Diagnostics are per-observation and not averaged; with several judges the last one wins.
        for observation in review.valid_observations:
            review.metadata.update(observation.reduced.metadata)
        if any(obs.status is ScoreStatus.FALLBACK for obs in review.valid_observations):
            review.status = ScoreStatus.FALLBACK

    def build_score(
        self,
        adapter: JudgeProtocol,
        review: JudgeReview,
        context: JudgeContext,
    ) -> Score:
        """Let the adapter shape the final score, then attach judge diagnostics."""
        score = adapter.finalize_judge_score(review, context)
        score.status = review.status
        if not review.status.is_usable and score.value:
            # The invariant this subsystem exists to protect: an unusable review is excluded from
            # the metric, never reported as a real number.
            logger.warning(
                f'{type(adapter).__name__} returned values {sorted(score.value)} for an unusable '
                f'judge review ({review.status.value}); dropping them.'
            )
            score.value = {}
        score.judge_detail = JudgeDetail(
            judge_models=[judge.model_id for judge in self.judges],
            valid_observations=len(review.valid_observations),
            total_observations=len(review.observations),
            failures=review.failure_counts,
            error=review.error,
        )
        if self.config.save_io:
            # Keeps the raw judge text inspectable in the reviews file, as ``Score.explanation``
            # used to before adapters stopped seeing responses.
            score.metadata = dict(score.metadata or {})
            score.metadata['judge_attempts'] = [
                attempt.model_dump(exclude_none=True, exclude={'usage'}) for attempt in review.attempts
            ]
        return score


def _combine(values: Sequence[Dict[str, float]]) -> Dict[str, float]:
    """Average per-metric dicts, only over the keys each entry actually carries."""
    buckets: Dict[str, List[float]] = {}
    for entry in values:
        for name, value in entry.items():
            buckets.setdefault(name, []).append(float(value))
    return {name: sum(items) / len(items) for name, items in buckets.items()}


def _jsonable(value: Any) -> Any:
    return value.model_dump() if isinstance(value, BaseModel) else value


__all__: Sequence[str] = ('JudgeExecutor', 'JudgeExecutorConfig')
