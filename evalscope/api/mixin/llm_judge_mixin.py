import threading
from functools import lru_cache
from pydantic import BaseModel, Field, create_model
from typing import TYPE_CHECKING, Any, List, Literal, Optional, Tuple, Type

from evalscope.api.evaluator import TaskState
from evalscope.api.metric import Score
from evalscope.constants import JudgeScoreType, JudgeStrategy, ScoreStatus, ScoringPolicy
from evalscope.metrics import LLMJudge
from evalscope.utils.argument_utils import get_secret_value
from evalscope.utils.deprecation_utils import deprecated_warning
from evalscope.utils.logger import get_logger

if TYPE_CHECKING:
    from evalscope.api.benchmark import BenchmarkMeta
    from evalscope.api.judge import (
        CaseVerdict,
        JudgeCase,
        JudgeContext,
        JudgeExecutor,
        JudgeRequest,
        JudgeReview,
        OutputContract,
        ReducedVerdict,
    )
    from evalscope.config import TaskConfig

logger = get_logger()


class _RatingVerdict(BaseModel):
    """The ``numeric`` judge contract: a reference-free 0-1 rating."""

    reasoning: str = ''
    score: float = Field(ge=0.0, le=1.0)


@lru_cache(maxsize=None)
def _correctness_model(labels: Tuple[str, ...]) -> Type[BaseModel]:
    """The ``pattern`` judge contract, whose allowed labels are the judge's ``score_mapping`` keys."""
    return create_model('CorrectnessVerdict', reasoning=(str, ''), verdict=(Literal[labels], ...))


class LLMJudgeMixin:
    """
    Mixin class for LLM Judge functionality.
    """

    scoring_policy: ScoringPolicy = ScoringPolicy.RULE_DEFAULT
    """Declares what this benchmark's rule and judge paths can do. See :class:`ScoringPolicy`."""

    llm_judge_default: Optional[bool] = None
    """[Deprecated] Superseded by :attr:`scoring_policy`. Will be removed in v2.0.0."""

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Map a legacy ``llm_judge_default`` declaration onto :attr:`scoring_policy`.

        ``True`` maps to ``JUDGE_DEFAULT`` rather than ``JUDGE_ONLY`` so a third-party adapter
        keeps rule scoring permitted.
        """
        super().__init_subclass__(**kwargs)
        legacy = cls.__dict__.get('llm_judge_default')
        if legacy is None or 'scoring_policy' in cls.__dict__:
            return
        deprecated_warning(
            logger, f'{cls.__name__} declares `llm_judge_default`, which is deprecated and will be removed in '
            'v2.0.0. Declare `scoring_policy` instead.'
        )
        cls.scoring_policy = ScoringPolicy.JUDGE_DEFAULT if legacy else ScoringPolicy.RULE_DEFAULT

    def __init__(self, benchmark_meta: 'BenchmarkMeta', task_config: Optional['TaskConfig'] = None):
        self._benchmark_meta = benchmark_meta
        self._task_config = task_config

        self._llm_judges: Optional[List[LLMJudge]] = None
        self._judge_executor: Optional['JudgeExecutor'] = None
        self._judge_executor_lock = threading.Lock()

        super().__init__()

    @property
    def llm_judges(self) -> List[LLMJudge]:
        """Every configured judge model, lazily built."""
        if self._llm_judges is None and self.use_llm_judge:
            self._llm_judges = self.init_llm_judges()
        return self._llm_judges or []

    @property
    def llm_judge(self) -> Optional[LLMJudge]:
        """The primary judge: the first one when several are configured."""
        judges = self.llm_judges
        return judges[0] if judges else None

    @llm_judge.setter
    def llm_judge(self, value: Optional[LLMJudge]):
        """Replace the configured judges with a single one."""
        self._llm_judges = [value] if value is not None else None
        # The executor is memoized around its judges, so it must not outlive them.
        self._judge_executor = None

    @property
    def judge_strategy(self) -> str:
        """Get the judge strategy from the task configuration."""
        return self._task_config.judge.strategy

    @property
    def use_llm_judge(self) -> bool:
        """Check if LLM judge is enabled."""
        if self.judge_strategy == JudgeStrategy.RULE:
            return False
        elif self.judge_strategy in (JudgeStrategy.LLM, JudgeStrategy.LLM_RECALL):
            return True
        elif self.judge_strategy == JudgeStrategy.AUTO:
            return self.scoring_policy.judge_by_default
        else:
            logger.warning(f'Unknown judge strategy: {self.judge_strategy}. Defaulting to False.')
            return False

    def validate_judge_strategy(self) -> None:
        """Reject a judge configuration this benchmark cannot honour, before any model call.

        Without these checks the run fails at scoring time -- after the samples have been
        generated -- or silently reports an all-zero result.
        """
        if self._task_config is None:
            return
        strategy = self.judge_strategy
        name = self._benchmark_meta.name
        if strategy in (JudgeStrategy.RULE, JudgeStrategy.LLM_RECALL) and not self.scoring_policy.rule_supported:
            raise ValueError(
                f"Benchmark '{name}' has no usable rule-based scoring, so judge_strategy='{strategy}' "
                'cannot produce a meaningful score. '
                "Use judge.strategy='auto' or 'llm' with judge.models."
            )
        if self.use_llm_judge and not self._task_config.judge.models:
            raise ValueError(
                f"Benchmark '{name}' scores with an LLM judge under judge_strategy='{strategy}', "
                'so judge.models must be provided.'
            )

    def _judge_specs(self) -> List[dict]:
        """Return typed judge model specifications as model-construction dictionaries."""
        contract = self._task_config.judge.contract.model_dump(exclude_none=True)
        return [{
            **spec.model_dump(exclude={'judge_id'}, exclude_none=True),
            **contract,
        } for spec in self._task_config.judge.models]

    def init_llm_judges(self) -> List[LLMJudge]:
        """Build every configured judge model.

        Returns:
            List[LLMJudge]: One judge per entry in ``judge.models``, empty for rule scoring.
        """
        if self.judge_strategy == JudgeStrategy.RULE:
            return []
        if not self._task_config.judge.models:
            raise ValueError(
                'LLM judge model arguments must be provided for LLM-based judge strategies. '
                'Please check your task configuration.'
            )
        judges = [LLMJudge(**get_secret_value(spec)) for spec in self._judge_specs()]
        for judge, spec in zip(judges, self._task_config.judge.models):
            judge.judge_id = spec.judge_id
        return judges

    # ------------------------------------------------------------------
    # Declarative judge path (evalscope.api.judge)
    # ------------------------------------------------------------------

    supports_position_swap: bool = False
    """Whether this benchmark has a meaningful pairwise placement dimension."""

    uses_pairwise_outcome: bool = False
    """Whether the adapter emits semantic pairwise outcomes instead of only swapped numeric scores."""

    official_position_swap: bool = False
    """Whether the benchmark's official protocol judges both placements by default."""

    @property
    def judge_executor(self) -> 'JudgeExecutor':
        """Lazily build the one executor that owns every judge call for this benchmark."""
        from evalscope.api.judge import JudgeExecutor, JudgeExecutorConfig

        if self._judge_executor is None:
            with self._judge_executor_lock:
                if self._judge_executor is None:
                    self._judge_executor = JudgeExecutor(
                        self.llm_judges,
                        JudgeExecutorConfig(
                            repeats=self._task_config.judge.repeats if self._task_config else 1,
                            position_swap=self._resolved_position_swap(),
                            aggregation=self._task_config.judge.aggregation if self._task_config else 'mean',
                            min_valid_judges=self._task_config.judge.min_valid_judges if self._task_config else 1,
                        ),
                    )
        return self._judge_executor

    def _resolved_position_swap(self) -> bool:
        """Resolve the user override against the benchmark's official position policy."""
        configured = self._task_config.judge.position_swap if self._task_config else 'auto'
        if configured == 'auto':
            return bool(self.official_position_swap)
        if not self.supports_position_swap:
            if configured == 'on':
                logger.warning(
                    f"Benchmark '{self._benchmark_meta.name}' has no pairwise position dimension; "
                    "ignoring judge.position_swap='on'."
                )
            return False
        return configured == 'on'

    def score_with_judge_contracts(
        self,
        original_prediction: str,
        filtered_prediction: str,
        reference: str,
        task_state: TaskState,
    ) -> Score:
        """Run this sample through the executor. Adapters do not override this."""
        from evalscope.api.judge import JudgeContext

        precomputed = self.pre_judge_score(original_prediction, filtered_prediction, reference, task_state)
        if precomputed is not None:
            return precomputed

        context = JudgeContext(
            task_state=task_state,
            original_prediction=original_prediction,
            filtered_prediction=filtered_prediction,
            reference=reference,
        )
        executor = self.judge_executor
        review = executor.execute(self, context)
        score = executor.build_score(self, review, context)
        return score

    def pre_judge_score(
        self,
        original_prediction: str,
        filtered_prediction: str,
        reference: str,
        task_state: TaskState,
    ) -> Optional[Score]:
        """Return an official rule-only short-circuit, or ``None`` to run the judge contract."""
        return None

    # Default judge hooks. A benchmark with one verdict per sample relies on these and overrides
    # only ``judge_prompt`` to change wording. Benchmarks with multiple cases or non-binary
    # scoring override the hooks instead.

    judge_metric_name: str = 'acc'
    """Metric key the default single-verdict ``reduce_judge_verdicts`` writes."""

    def judge_prompt(self, context: 'JudgeContext') -> str:
        """The default judge prompt for one sample, without any format instruction."""
        return self.llm_judge.build_prompt(
            pred=context.original_prediction,
            gold=context.reference,
            question=context.task_state.input_text,
        )

    def default_judge_contract(self) -> 'OutputContract':
        """The built-in contract the judge's ``score_type`` selects."""
        from evalscope.api.judge import OutputContract

        if self.llm_judge.score_type == JudgeScoreType.NUMERIC:
            return OutputContract(schema_model=_RatingVerdict)
        return OutputContract(schema_model=_correctness_model(tuple(sorted(self.llm_judge.score_mapping))))

    def build_judge_cases(self, context: 'JudgeContext') -> List['JudgeCase']:
        """One verdict case by default."""
        from evalscope.api.judge import JudgeCase

        return [JudgeCase(case_id='match', output_contract=self.default_judge_contract())]

    def build_judge_request(
        self,
        case: 'JudgeCase',
        placement: Any,
        completed_cases: List['CaseVerdict'],
        context: 'JudgeContext',
    ) -> 'JudgeRequest':
        """Render the default correctness case into a single judge message."""
        from evalscope.api.judge import JudgeRequest
        from evalscope.api.messages import ChatMessageSystem, ChatMessageUser

        prompt = self.judge_prompt(context) + case.output_contract.instruction()
        messages: List[Any] = []
        system = getattr(self.llm_judge, 'system_prompt', None)
        if system:
            messages.append(ChatMessageSystem(content=system))
        messages.append(ChatMessageUser(content=prompt))
        return JudgeRequest(messages=messages)

    def reduce_judge_verdicts(
        self,
        case_verdicts: List['CaseVerdict'],
        context: 'JudgeContext',
    ) -> 'ReducedVerdict':
        """Fold the default single verdict into ``{judge_metric_name: value}``."""
        from evalscope.api.judge import ReducedVerdict

        if not case_verdicts:
            return ReducedVerdict()
        verdict = case_verdicts[0].value
        if self.llm_judge.score_type == JudgeScoreType.NUMERIC:
            value = float(verdict.score)
        else:
            value = float(self.llm_judge.score_mapping[verdict.verdict])
        return ReducedVerdict(value={self.judge_metric_name: value})

    def expand_judge_cases(
        self,
        stage: int,
        completed_cases: List['CaseVerdict'],
        context: 'JudgeContext',
    ) -> List['JudgeCase']:
        """No derived cases by default; only staged benchmarks override this."""
        return []

    def judge_fallback_verdict(self, case: 'JudgeCase', context: 'JudgeContext') -> Optional['CaseVerdict']:
        """No rule fallback by default: an unanswerable case excludes the sample."""
        return None

    def finalize_judge_score(self, review: 'JudgeReview', context: 'JudgeContext') -> Score:
        """Wrap the aggregated values in a ``Score`` carrying the sample's predictions."""
        return Score(
            extracted_prediction=context.filtered_prediction,
            prediction=context.original_prediction,
            value=dict(review.value),
            metadata={
                'source': 'llm_judge',
                'judge_strategy': self.judge_strategy,
                'model': getattr(self.llm_judge, 'judge_id', self.llm_judge.model_id),
                'non_official_position_swap': self._task_config is not None
                and self._resolved_position_swap() != self.official_position_swap,
                **review.metadata,
            },
        )

    def fallback_to_rule_score(self, rule_based_score: Score, judge_score: Score) -> Score:
        """Retain rule evidence when a ``JUDGE_DEFAULT`` review is unavailable."""
        fallback = rule_based_score.model_copy(deep=True)
        fallback.status = ScoreStatus.DEGRADED
        fallback.judge_summary = judge_score.judge_summary.model_copy(
            update={'status': ScoreStatus.DEGRADED}
        ) if judge_score.judge_summary is not None else None
        fallback.metadata = {
            **(fallback.metadata or {}),
            **(judge_score.metadata or {}),
            'judge_unavailable': judge_score.status.value,
        }
        return fallback

    def _merge_scores(self, rule_based_score: Score, llm_score: Score) -> Score:
        """Merge the rule score with the judge score for the LLM_RECALL strategy.

        ``llm_recall`` exists to recover rule-based misses, so the judge can only raise the
        score: the result is ``max(rule, judge)``. A failed judge must not erase rule evidence.
        """
        if not llm_score.status.is_usable or not llm_score.value:
            # The rule score stands, so the sample is still scored -- it just fell back.
            rule_based_score.status = ScoreStatus.FALLBACK
            rule_based_score.judge_summary = llm_score.judge_summary.model_copy(
                update={'status': ScoreStatus.FALLBACK}
            ) if llm_score.judge_summary is not None else None
            rule_based_score.metadata = {
                **(rule_based_score.metadata or {}),
                'judge_unavailable': llm_score.status.value,
            }
            return rule_based_score

        rule_value = float(rule_based_score.main_value or 0.0)
        judge_value = float(llm_score.main_value or 0.0)
        rule_based_score.main_value = max(rule_value, judge_value)
        rule_based_score.explanation = llm_score.explanation
        rule_based_score.metadata = {**(rule_based_score.metadata or {}), **(llm_score.metadata or {})}
        rule_based_score.status = llm_score.status
        rule_based_score.judge_summary = llm_score.judge_summary

        return rule_based_score
