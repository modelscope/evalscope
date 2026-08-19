import os
from typing import Any, Dict, List, Optional

from evalscope.api.messages import ChatMessage, ChatMessageSystem, ChatMessageUser
from evalscope.constants import EvalType, JudgeScoreType
from evalscope.utils.deprecation_utils import deprecated_warning
from evalscope.utils.logger import get_logger
from .base import BaseJudge

logger = get_logger()

# Sentinel that ``judge`` returns instead of raising on a failed request. Consumers must fail
# closed on it: an ``[ERROR]`` string must never reach a parser, and never be scored.
JUDGE_ERROR_PREFIX = '[ERROR]'

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


class LLMJudge(BaseJudge):
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
        score_pattern: Optional[str] = None,
        score_mapping: Optional[Dict[str, float]] = None,
        score_type: str = JudgeScoreType.PATTERN,  # 'pattern', 'numeric'
        **kwargs
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
            score_pattern (str, optional): [Deprecated] No longer has any effect.
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
        if score_pattern:
            deprecated_warning(
                logger, 'The `score_pattern` judge parameter is deprecated and no longer has any effect: the '
                'judge now replies with a JSON object validated against a schema. It will be removed in v2.0.0.'
            )

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

    def judge(
        self,
        prompt: str = '',
        system_prompt: Optional[str] = None,
        messages: Optional[List[ChatMessage]] = None
    ) -> str:
        """
        Generate a response from the LLM based on the provided prompt and context.
        If messages is provided, it will be used as the input context.

        Args:
            prompt (str): The prompt to evaluate
            system_prompt (str, optional): The system prompt to use for the evaluation
            messages (List[ChatMessage], optional): A list of chat messages to include in the evaluation
        Returns:
            str: The response from the LLM
        """
        # parse messages
        if messages is not None:
            input_messages = messages
        else:
            system_content = system_prompt or self.system_prompt
            input_messages = [ChatMessageUser(content=prompt)]
            if system_content:
                input_messages.insert(0, ChatMessageSystem(content=system_content))
        try:
            # Send request using ServerModelAdapter
            response = self.model.generate(input_messages)

            # Extract content from response
            llm_response = response.completion
            return llm_response
        except Exception as e:
            error_message = f'Error occurred during {self.model_id}@{self.api_url} LLM judge evaluation: {e}'
            logger.error(error_message)
            return f'{JUDGE_ERROR_PREFIX} {error_message}'

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
