import os
from typing import Any, Dict, List, Optional

from evalscope.api.messages import ChatMessage
from evalscope.api.model import ModelOutput
from evalscope.constants import EvalType, JudgeScoreType
from evalscope.utils.logger import get_logger

logger = get_logger()

# The templates state the grading criteria only. The reply format is appended by
# ``OutputContract.instruction()`` so the prompt and the parser cannot drift apart.
DEFAULT_PROMPT_TEMPLATE = """Your job is to look at a question, a gold target, and a predicted answer, and decide whether the predicted answer is correct or incorrect.

[Question]
{question}

[Reference Answer]
{gold}

[Predicted Answer]
{pred}

Evaluate the model's answer based on correctness compared to the reference answer.
Grade the predicted answer of this new question as one of:
A: CORRECT
B: INCORRECT"""  # noqa: E501


DEFAULT_NUMERIC_SCORE_TEMPLATE = """Please act as an impartial judge and evaluate the quality of the response provided by an AI assistant to the user question displayed below. Your evaluation should consider factors such as the helpfulness, relevance, accuracy, depth, creativity, and level of detail of the response.
Begin your evaluation by providing a short explanation. Be as objective as possible.
Rate the response on a scale of 0 (worst) to 1 (best).

[Question]
{question}

[Response]
{pred}"""  # noqa: E501

DEFAULT_JUDGE_MODEL = 'Qwen/Qwen3-235B-A22B'
DEFAULT_API_URL = 'https://api-inference.modelscope.cn/v1/'


class LLMJudge:
    """
    A metric that uses LLM to judge the quality of model predictions by comparing them with reference answers.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        api_url: Optional[str] = None,
        model_id: Optional[str] = None,
        eval_type: Optional[str] = None,
        model_args: Optional[Dict[str, Any]] = None,
        system_prompt: Optional[str] = None,
        prompt_template: Optional[str] = None,
        generation_config: Optional[Dict[str, Any]] = None,
        score_mapping: Optional[Dict[str, float]] = None,
        score_type: str = JudgeScoreType.PATTERN,  # 'pattern', 'numeric'
    ):
        """
        Initialize LLMJudge metric.

        Args:
            api_key (str, optional): API key for OpenAI or compatible service
            api_url (str, optional): API base URL
            model_id (str, optional): Model ID for LLM
            eval_type (str, optional): Evaluation LLM type for the judge
            model_args (dict, optional): Additional model arguments for the judge
            system_prompt (str, optional): System prompt for the judge
            prompt_template (str, optional): Prompt template for the judge
            generation_config (dict, optional): Generation configuration for the judge
            score_mapping (dict, optional): Allowed verdict labels mapped to their score values
            score_type (str, optional): Which built-in judge contract to use, defaults to 'pattern'.
                - 'pattern': grade against a reference answer, choosing one of the score_mapping labels
                - 'numeric': rate the response on a 0-1 scale without a reference
        """
        self.api_key = api_key or os.environ.get('MODELSCOPE_SDK_TOKEN', 'EMPTY')
        self.api_url = api_url or os.environ.get('MODELSCOPE_API_BASE', DEFAULT_API_URL)
        self.model_id = model_id or os.environ.get('MODELSCOPE_JUDGE_LLM', DEFAULT_JUDGE_MODEL)
        self.eval_type = eval_type or EvalType.OPENAI_API
        self.system_prompt = system_prompt or os.environ.get('JUDGE_SYSTEM_PROMPT', None)
        self.generation_config = generation_config or {'temperature': 0.0, 'max_tokens': 4096}
        self.model_args = model_args or {}

        self.score_type = score_type
        if self.score_type == JudgeScoreType.NUMERIC:
            self.prompt_template = prompt_template or os.environ.get(
                'JUDGE_PROMPT_TEMPLATE', DEFAULT_NUMERIC_SCORE_TEMPLATE
            )
        elif self.score_type == JudgeScoreType.PATTERN:
            self.prompt_template = prompt_template or os.environ.get('JUDGE_PROMPT_TEMPLATE', DEFAULT_PROMPT_TEMPLATE)
        else:
            raise ValueError(f"Invalid score_type: {self.score_type}. Must be 'pattern' or 'numeric'.")

        self.score_mapping = score_mapping or {'A': 1.0, 'B': 0.0}

        self._init_server_adapter()

    def _init_server_adapter(self) -> None:
        from evalscope.api.model import GenerateConfig, get_model

        self.model = get_model(
            model=self.model_id,
            eval_type=self.eval_type,
            base_url=self.api_url,
            api_key=self.api_key,
            config=GenerateConfig(**self.generation_config),
            model_args=self.model_args,
        )

    def generate(self, messages: List[ChatMessage]) -> ModelOutput:
        """Run one judge request and preserve the provider response unchanged.

        Transport failures deliberately propagate to the executor, where they become typed
        ``transport_error`` attempts.  Returning a magic string here used to make failures look
        like malformed judge verdicts or, worse, valid zero scores.
        """
        return self.model.generate(messages)

    def build_prompt(self, pred: str, gold: str, question: Optional[str] = None) -> str:
        if question is None:
            question = 'Not provided'

        # check variables in prompt_template
        prompt = self.prompt_template
        if '{question}' in self.prompt_template:
            prompt = prompt.replace('{question}', question)
        if '{pred}' in self.prompt_template:
            prompt = prompt.replace('{pred}', pred)
        if '{gold}' in self.prompt_template:
            prompt = prompt.replace('{gold}', gold)
        return prompt
