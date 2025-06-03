import os
import re
from typing import Any, Dict, List, Optional

from evalscope.utils.logger import get_logger

logger = get_logger()

DEFAULT_PROMPT_TEMPLATE = """Your job is to look at a question, a gold target, and a predicted answer, and return a letter "A" or "B" to indicate whether the predicted answer is correct or incorrect.

Question: {question}

Reference Answer: {gold}

Model Answer: {pred}

Evaluate the model's answer based on correctness compared to the reference answer.
Grade the predicted answer of this new question as one of:
A: CORRECT
B: INCORRECT

Just return the letters "A" or "B", with no text around it.
"""  # noqa: E501

DEFAULT_JUDGE_MODEL = 'Qwen/Qwen3-235B-A22B'
DEFAULT_API_URL = 'https://api-inference.modelscope.cn/v1/'


class LLMJudge:
    """
    A metric that uses LLM to judge the quality of model predictions by comparing them with reference answers.
    """

    def __init__(self,
                 api_key: Optional[str] = None,
                 api_url: Optional[str] = None,
                 model_id: Optional[str] = None,
                 system_prompt: Optional[str] = None,
                 prompt_template: Optional[str] = None,
                 generation_config: Optional[Dict[str, Any]] = None,
                 **kwargs):
        """
        Initialize LLMJudge metric.

        Args:
            api_key (str, optional): API key for OpenAI or compatible service
            api_base (str, optional): API base URL
            model_id (str, optional): Model ID for LLM
            system_prompt (str, optional): System prompt for the judge
            prompt_template (str, optional): Prompt template for the judge
            generation_config (dict, optional): Generation configuration for the judge
        """
        self.api_key = api_key or os.environ.get('MODELSCOPE_SDK_TOKEN', 'EMPTY')
        self.api_url = api_url or os.environ.get('MODELSCOPE_API_BASE', DEFAULT_API_URL)
        self.model_id = model_id or os.environ.get('MODELSCOPE_JUDGE_LLM', DEFAULT_JUDGE_MODEL)
        self.system_prompt = system_prompt or os.environ.get('JUDGE_SYSTEM_PROMPT', None)
        self.prompt_template = prompt_template or os.environ.get('JUDGE_PROMPT_TEMPLATE', DEFAULT_PROMPT_TEMPLATE)
        self.generation_config = generation_config or {}

        from evalscope.models import ServerModelAdapter

        # Initialize ServerModelAdapter
        self.server_adapter = ServerModelAdapter(api_url=self.api_url, model_id=self.model_id, api_key=self.api_key)

    def __call__(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """
        Args:
            prompt (str): The prompt to evaluate
            system_prompt (str, optional): The system prompt to use for the evaluation
        Returns:
            str: The response from the LLM
        """
        input_data = {'data': [prompt], 'system_prompt': system_prompt or self.system_prompt}

        # Inference configuration
        infer_cfg = {'temperature': 0.0, 'max_tokens': 1024}
        if self.generation_config:
            infer_cfg.update(self.generation_config)

        if self.model_id == DEFAULT_JUDGE_MODEL:
            # Disable thinking for the default judge model
            infer_cfg['enable_thinking'] = self.generation_config.get('enable_thinking', False)

        try:
            # Send request using ServerModelAdapter
            response = self.server_adapter.process_single_input(input_data, infer_cfg)

            # Extract content from response
            llm_response = response.get('choices', [{}])[0].get('message', {}).get('content', '')
            return llm_response
        except Exception as e:
            logger.error(f'Error occurred during {self.model_id}@{self.api_url} LLM judge evaluation: {e}')
            return ''

    def build_prompt(self, pred: str, gold: str, question: Optional[str] = None):
        if question is None:
            question = 'Not provided'
        return self.prompt_template.format(question=question, pred=pred, gold=gold)

    def get_score(self, response: str) -> float:
        if response is None:
            return 0
        match = re.search(r'(A|B)', response)
        if match:
            answer = match.group(0)
            if answer == 'A':
                return 1
            elif answer == 'B':
                return 0
        else:
            return 0
