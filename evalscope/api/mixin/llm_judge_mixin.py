import threading
from pydantic import BaseModel
from typing import TYPE_CHECKING, Any, List, Literal, Optional

from evalscope.api.evaluator import TaskState
from evalscope.api.metric import Score
from evalscope.constants import JudgeStrategy, ScoreStatus, ScoringPolicy
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
        ReducedVerdict,
    )
    from evalscope.config import TaskConfig

logger = get_logger()

# The default correctness prompt, reproducing the legacy A/B judge without its "return only the
# letter" tail so the output-contract's JSON instruction can be appended cleanly.
_DEFAULT_CORRECTNESS_PROMPT = """Your job is to look at a question, a gold target, and a predicted \
answer, and decide whether the predicted answer is correct.

[Question]
{question}

[Reference Answer]
{gold}

[Predicted Answer]
{pred}

Grade the predicted answer as one of:
A: CORRECT
B: INCORRECT"""


class _CorrectnessVerdict(BaseModel):
    reasoning: str = ''
    verdict: Literal['A', 'B']


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
        keeps today's behaviour: ``auto`` uses the judge and rule scoring stays permitted.
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

        self._llm_judge: Optional[LLMJudge] = None
        """LLM judge instance"""

        self._judge_executor: Optional['JudgeExecutor'] = None
        self._judge_executor_lock = threading.Lock()

        super().__init__()

    @property
    def llm_judge(self) -> Optional[LLMJudge]:
        """Get LLM judge instance with lazy initialization."""
        if self._llm_judge is None and self.use_llm_judge:
            self._llm_judge = self.init_llm_judge()
        return self._llm_judge

    @llm_judge.setter
    def llm_judge(self, value: Optional[LLMJudge]):
        """Set LLM judge instance."""
        self._llm_judge = value

    @property
    def judge_strategy(self) -> str:
        """Get the judge strategy from the task configuration."""
        return self._task_config.judge_strategy

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

        ``rule`` and ``llm_recall`` both need a usable rule score, so neither can run on a
        ``JUDGE_ONLY`` benchmark. Without these checks the run fails at scoring time -- after the
        samples have been generated -- or silently reports an all-zero result.
        """
        if self._task_config is None:
            return
        strategy = self.judge_strategy
        name = self._benchmark_meta.name
        if strategy in (JudgeStrategy.RULE, JudgeStrategy.LLM_RECALL) and not self.scoring_policy.rule_supported:
            raise ValueError(
                f"Benchmark '{name}' has no usable rule-based scoring, so judge_strategy='{strategy}' "
                'cannot produce a meaningful score. '
                "Use judge_strategy='auto' or 'llm' with judge_model_args."
            )
        if self.use_llm_judge and not self._task_config.judge_model_args:
            raise ValueError(
                f"Benchmark '{name}' scores with an LLM judge under judge_strategy='{strategy}', "
                'so judge_model_args must be provided.'
            )

    def init_llm_judge(self) -> Optional[LLMJudge]:
        """
        Initialize the LLM judge for the benchmark.

        Returns:
            Optional[LLMJudge]: The initialized LLM judge instance or None
        """

        if self.judge_strategy == JudgeStrategy.RULE:
            return None
        else:
            if not self._task_config.judge_model_args:
                raise ValueError(
                    'LLM judge model arguments must be provided for LLM-based judge strategies. '
                    'Please check your task configuration.'
                )
            judge_model_args = get_secret_value(self._task_config.judge_model_args)
            return LLMJudge(**judge_model_args)

    # ------------------------------------------------------------------
    # Declarative judge path (evalscope.api.judge)
    # ------------------------------------------------------------------

    uses_judge_contracts: bool = False
    """Set by an adapter that declares :class:`JudgeCase` objects instead of calling the judge."""

    judge_position_swap: bool = False
    """Ask each case in both orders and treat the pair as one atomic observation."""

    @property
    def judge_executor(self) -> 'JudgeExecutor':
        """Lazily build the one executor that owns every judge call for this benchmark."""
        from evalscope.api.judge import JudgeExecutor, JudgeExecutorConfig

        if self._judge_executor is None:
            with self._judge_executor_lock:
                if self._judge_executor is None:
                    self._judge_executor = JudgeExecutor(
                        [self.llm_judge],
                        JudgeExecutorConfig(position_swap=self.judge_position_swap),
                    )
        return self._judge_executor

    def score_with_judge_contracts(
        self,
        original_prediction: str,
        filtered_prediction: str,
        reference: str,
        task_state: TaskState,
    ) -> Score:
        """Run this sample through the executor. Adapters do not override this."""
        from evalscope.api.judge import JudgeContext

        context = JudgeContext(
            task_state=task_state,
            original_prediction=original_prediction,
            filtered_prediction=filtered_prediction,
            reference=reference,
        )
        executor = self.judge_executor
        review = executor.execute(self, context)
        return executor.build_score(self, review, context)

    # Default judge hooks. A benchmark with a single correct/incorrect verdict per sample can turn
    # on ``uses_judge_contracts`` and rely on these defaults, overriding only ``judge_prompt`` to
    # change wording. Benchmarks with multiple cases or non-binary scoring override the hooks.

    judge_metric_name: str = 'acc'
    """Metric key the default single-verdict ``reduce_judge_verdicts`` writes."""

    def judge_prompt(self, context: 'JudgeContext') -> str:
        """The default correctness prompt for one sample, without any format instruction."""
        return _DEFAULT_CORRECTNESS_PROMPT.format(
            question=context.task_state.input_text,
            gold=context.reference,
            pred=context.original_prediction,
        )

    def build_judge_cases(self, context: 'JudgeContext') -> List['JudgeCase']:
        """One binary correctness case by default."""
        from evalscope.api.judge import JudgeCase, OutputContract

        return [JudgeCase(case_id='match', output_contract=OutputContract(schema_model=_CorrectnessVerdict))]

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
        """Fold the default single binary verdict into ``{judge_metric_name: 1.0|0.0}``."""
        from evalscope.api.judge import ReducedVerdict

        correct = 1.0 if case_verdicts and case_verdicts[0].value.verdict == 'A' else 0.0
        return ReducedVerdict(value={self.judge_metric_name: correct})

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
                'model': self.llm_judge.model_id,
                **review.metadata,
            },
        )

    def maybe_llm_match_score(
        self,
        original_prediction: str,
        filtered_prediction: str,
        reference: str,
        task_state: TaskState,
        rule_based_score: Optional[Score] = None,
    ) -> Score:
        """
        Compute the match score between the original and filtered predictions against the reference.

        Args:
            original_prediction: The original prediction output from the model.
            filtered_prediction: The filtered prediction output from the model.
            reference: The ground truth reference output.
            task_state: The current task state.
            original_score: Optional original score to be used for comparison.

        Returns:
            Score: The computed match score.
        """
        # If LLM judge is not used, return the rule-based score directly
        if not self.use_llm_judge:
            return rule_based_score

        # A perfect rule score cannot be raised, so asking the judge would only cost money.
        # The threshold is exact because ``_merge_scores`` takes the maximum.
        if float(rule_based_score.main_value) >= 1.0:
            return rule_based_score

        # Compute LLM judge score
        llm_score = self.llm_match_score(
            original_prediction=original_prediction,
            filtered_prediction=filtered_prediction,
            reference=reference,
            task_state=task_state,
        )

        # For LLM RECALL, merge the scores
        return self._merge_scores(rule_based_score, llm_score)

    def llm_match_score(
        self,
        original_prediction: str,
        filtered_prediction: str,
        reference: str,
        task_state: TaskState,
    ) -> Score:
        """Compute the LLM match score.

        Args:
            original_prediction (str): The original prediction output from the model.
            filtered_prediction (str): The filtered prediction output from the model.
            reference (str): The ground truth reference output.
            task_state (TaskState): The current task state.

        Returns:
            Score: The computed match score.
        """
        if self.uses_judge_contracts:
            return self.score_with_judge_contracts(
                original_prediction=original_prediction,
                filtered_prediction=filtered_prediction,
                reference=reference,
                task_state=task_state,
            )

        score = Score(
            extracted_prediction=filtered_prediction,
            prediction=original_prediction,
        )

        question = task_state.input_text

        # Request judge and obtain score
        prompt = self.llm_judge.build_prompt(pred=original_prediction, gold=reference, question=question)
        judge_response = self.llm_judge.judge(prompt)
        judge_score = self.llm_judge.get_score(judge_response)

        score.value = {'acc': judge_score}
        score.explanation = f'LLM judge: {judge_response}'
        score.metadata = {
            'source': 'llm_judge',
            'judge_strategy': self.judge_strategy,
            'model': self.llm_judge.model_id
        }

        return score

    def _merge_scores(self, rule_based_score: Score, llm_score: Score) -> Score:
        """Merge the rule score with the judge score for the LLM_RECALL strategy.

        ``llm_recall`` exists to recover rule-based misses, so the judge can only raise the
        score: the result is ``max(rule, judge)``. A judge that produced no usable value leaves
        the rule score untouched -- a failed judge must not erase rule evidence, which is what
        happened while ``[ERROR]`` responses were being scored as 0.
        """
        if not llm_score.status.is_usable or not llm_score.value:
            # The rule score stands, so the sample is still scored -- it just fell back.
            rule_based_score.status = ScoreStatus.FALLBACK
            rule_based_score.judge_detail = llm_score.judge_detail
            rule_based_score.metadata = {
                **(rule_based_score.metadata or {}),
                'judge_unavailable': llm_score.status.value,
            }
            return rule_based_score

        rule_value = float(rule_based_score.main_value or 0.0)
        judge_value = float(llm_score.main_value or 0.0)
        rule_based_score.main_value = max(rule_value, judge_value)
        rule_based_score.explanation = llm_score.explanation
        rule_based_score.metadata = llm_score.metadata
        rule_based_score.status = llm_score.status
        rule_based_score.judge_detail = llm_score.judge_detail

        return rule_based_score
