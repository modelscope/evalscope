from typing import TYPE_CHECKING, Any, Dict, List, Optional

from evalscope.api.evaluator import TaskState
from evalscope.api.metric import Score
from evalscope.constants import JudgeStrategy
from evalscope.metrics import LLMJudge
from evalscope.utils.logger import get_logger

if TYPE_CHECKING:
    from evalscope.api.benchmark import BenchmarkMeta
    from evalscope.config import TaskConfig

logger = get_logger()


class LLMJudgeMixin:
    """
    Mixin class for LLM Judge functionality.
    """

    def __init__(self, benchmark_meta: 'BenchmarkMeta', task_config: Optional['TaskConfig'] = None):
        self._benchmark_meta = benchmark_meta
        self._task_config = task_config

        self._use_llm_judge = False
        """Whether to use LLM as a judge"""

        self._llm_judge: Optional[LLMJudge] = None
        """LLM judge instance"""

        super().__init__(benchmark_meta=benchmark_meta, task_config=task_config)

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
        elif self.judge_strategy == JudgeStrategy.LLM:
            return True
        elif self.judge_strategy == JudgeStrategy.LLM_RECALL:
            return True
        elif self.judge_strategy == JudgeStrategy.AUTO:
            return self._use_llm_judge
        else:
            logger.warning(f'Unknown judge strategy: {self.judge_strategy}. Defaulting to False.')
            return False

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
            return LLMJudge(**self._task_config.judge_model_args)

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

        # For LLM_RECALL, if rule-based score is already perfect, skip LLM judge
        if float(rule_based_score.main_value) > 0.99:
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
        """
        Merge rule-based score with LLM judge score for LLM_RECALL strategy.

        Args:
            rule_based_score: The original rule-based score
            llm_score: The LLM judge score

        Returns:
            Score: The merged score
        """
        # Update the main value with LLM judge result
        rule_based_score.main_value = llm_score.main_value
        rule_based_score.explanation = llm_score.explanation
        rule_based_score.metadata = llm_score.metadata

        return rule_based_score
