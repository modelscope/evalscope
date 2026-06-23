import os
from typing import Any, Dict, List, Optional

from evalscope.api.messages import ChatMessage, ChatMessageSystem, ChatMessageUser
from evalscope.constants import EvalType, JudgeScoreType
from evalscope.utils.logger import get_logger
from .base import BaseJudge
from .score_extractors import NumericScoreExtractor, PatternScoreExtractor, ScoreExtractor

logger = get_logger()

DEFAULT_PROMPT_TEMPLATE = """Your job is to look at a question, a gold target, and a predicted answer, and return a letter "A" or "B" to indicate whether the predicted answer is correct or incorrect.

[Question]
{question}

[Reference Answer]
{gold}

[Predicted Answer]
{pred}

Evaluate the model's answer based on correctness compared to the reference answer.
Grade the predicted answer of this new question as one of:
A: CORRECT
B: INCORRECT

Just return the letters "A" or "B", with no text around it.
"""  # noqa: E501


DEFAULT_NUMERIC_SCORE_TEMPLATE = """Please act as an impartial judge and evaluate the quality of the response provided by an AI assistant to the user question displayed below. Your evaluation should consider factors such as the helpfulness, relevance, accuracy, depth, creativity, and level of detail of the response.
Begin your evaluation by providing a short explanation. Be as objective as possible.
After providing your explanation, you must rate the response on a scale of 0 (worst) to 1 (best) by strictly following this format: \"[[rating]]\", for example: \"Rating: [[0.5]]\"

[Question]
{question}

[Response]
{pred}
"""  # noqa: E501

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
            score_pattern (str, optional): Regex pattern to extract score from LLM response
            score_mapping (dict, optional): Mapping from extracted score to float value
            score_type (str, optional): Type of score extraction strategy ('pattern', 'numeric') defaults to 'pattern'.
                - 'pattern': Use score_pattern and score_mapping to extract categorical scores
                - 'numeric': Treat the extracted value as a direct numerical score
        """
        self.api_key = api_key or os.environ.get('MODELSCOPE_SDK_TOKEN', 'EMPTY')
        self.api_url = api_url or os.environ.get('MODELSCOPE_API_BASE', DEFAULT_API_URL)
        self.model_id = model_id or os.environ.get('MODELSCOPE_JUDGE_LLM', DEFAULT_JUDGE_MODEL)
        self.eval_type = eval_type or EvalType.OPENAI_API
        self.system_prompt = system_prompt or os.environ.get('JUDGE_SYSTEM_PROMPT', None)
        self.generation_config = generation_config or {'temperature': 0.0, 'max_tokens': 4096}
        self.model_args = model_args or {}

        # Build score extractor based on score_type (Strategy pattern)
        self.score_type = score_type
        if self.score_type == JudgeScoreType.NUMERIC:
            pattern = score_pattern or r'\[\[(\d+(?:\.\d+)?)\]\]'
            self.prompt_template = prompt_template or os.environ.get(
                'JUDGE_PROMPT_TEMPLATE', DEFAULT_NUMERIC_SCORE_TEMPLATE
            )
            self._score_extractor: ScoreExtractor = NumericScoreExtractor(pattern=pattern)
        elif self.score_type == JudgeScoreType.PATTERN:
            # Anchor to only accept a standalone A or B (avoid false positives)
            pattern = score_pattern or r'^\s*([AB])\s*$'
            self.prompt_template = prompt_template or os.environ.get('JUDGE_PROMPT_TEMPLATE', DEFAULT_PROMPT_TEMPLATE)
            self._score_extractor = PatternScoreExtractor(pattern=pattern, score_mapping=score_mapping)
        else:
            raise ValueError(f"Invalid score_type: {self.score_type}. Must be 'pattern' or 'numeric'.")

        # Keep score_pattern and score_mapping as public attrs for backward compatibility
        self.score_pattern = pattern
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
            return f'[ERROR] {error_message}'

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

    def get_score(self, response: str) -> float:
        """
        Extract score from LLM response using the configured extractor strategy.

        Args:
            response (str): The response from the LLM

        Returns:
            float: The numeric score extracted from the response
        """
        if response is None:
            return 0.0
        return self._score_extractor.extract(response)
