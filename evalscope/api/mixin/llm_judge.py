from typing import TYPE_CHECKING, Any, Dict, List, Optional

from evalscope.api.evaluator import TaskState
from evalscope.api.metric import Score
from evalscope.constants import JudgeStrategy
from evalscope.metrics import LLMJudge
from evalscope.utils.logger import get_logger

if TYPE_CHECKING:
    from evalscope.config import TaskConfig

logger = get_logger()


class LLMJudgeMixin:
    """
    Mixin class for LLM Judge functionality.
    """

    def __init__(self, task_config: Optional['TaskConfig'] = None):
        self._task_config = task_config
        self._use_llm_judge = False
        """Whether to use LLM as a judge"""

        self._llm_judge: Optional[LLMJudge] = None

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
    def use_llm_judge(self) -> bool:
        """Check if LLM judge is enabled."""
        if self._task_config.judge_strategy == JudgeStrategy.RULE:
            return False
        elif self._task_config.judge_strategy == JudgeStrategy.LLM:
            return True
        elif self._task_config.judge_strategy == JudgeStrategy.AUTO:
            return self._use_llm_judge
        elif self._task_config.judge_strategy == JudgeStrategy.LLM_RECALL:
            return True
        else:
            logger.warning(f'Unknown judge strategy: {self._task_config.judge_strategy}. Defaulting to False.')
            return False

    def init_llm_judge(self) -> Optional[LLMJudge]:
        """
        Initialize the LLM judge for the benchmark.

        Returns:
            Optional[LLMJudge]: The initialized LLM judge instance or None
        """
        if self._task_config is None:
            logger.warning('Task config is not available for LLM judge initialization')
            return None

        if self._task_config.judge_strategy == JudgeStrategy.RULE:
            return None
        else:
            return LLMJudge(**self._task_config.judge_model_args)

    def enable_llm_judge(self) -> None:
        """Enable LLM judge functionality."""
        self.use_llm_judge = True
        self._llm_judge = None

    def disable_llm_judge(self) -> None:
        """Disable LLM judge functionality."""
        self.use_llm_judge = False
        self._llm_judge = None

    def llm_match_score(
        self,
        original_prediction: str,
        filtered_prediction: str,
        reference: str,
        task_state: TaskState,
        original_score: Optional[Score] = None,
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
        # TODO: Implement
        judge_response = self.llm_judge.judge(
            prompt=original_prediction, system_prompt=filtered_prediction, reference=reference, task_state=task_state)
        return judge_response
