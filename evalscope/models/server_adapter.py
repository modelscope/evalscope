import openai
from openai.types.chat import ChatCompletion
from typing import List, Optional, Union

from evalscope.models.base_adapter import BaseModelAdapter
from evalscope.utils.logger import get_logger

logger = get_logger()


class ServerModelAdapter(BaseModelAdapter):
    """
    Server model adapter to request remote API model and generate results.
    """

    def __init__(self, api_url: str, model_id: str, api_key: str = 'EMPTY', **kwargs):
        """
        Args:
            api_url: The URL of the remote API model.
            model_id: The ID of the remote API model.
            api_key: The API key of the remote API model.
        """
        self.api_url = api_url.rstrip('/').rsplit('/chat/completions', 1)[0]
        self.model_id = model_id
        self.api_key = api_key

        self.client = openai.OpenAI(
            api_key=api_key,
            base_url=self.api_url,
        )

        self.seed = kwargs.get('seed', None)
        self.timeout = kwargs.get('timeout', 60)
        self.model_cfg = {'api_url': api_url, 'model_id': model_id, 'api_key': api_key}
        super().__init__(model=None, model_cfg=self.model_cfg, **kwargs)

    def predict(self, inputs: List[Union[str, dict, list]], infer_cfg: dict = None) -> List[dict]:
        """
        Model prediction func.

        Args:
            inputs (List[Union[str, dict, list]]): The input data.
            infer_cfg (dict): Inference configuration.

        Returns:
            res (List[dict]): The model prediction results.
        """
        infer_cfg = infer_cfg or {}
        results = []

        for input_item in inputs:
            response = self.process_single_input(input_item, infer_cfg)
            results.append(response)

        return results

    def process_single_input(self, input_item: dict, infer_cfg: dict) -> dict:
        """Process a single input item."""
        data: list = input_item['data']
        if isinstance(data[0], tuple):  # for truthful_qa and hellaswag
            query = '\n'.join(''.join(item) for item in data)
            system_prompt = input_item.get('system_prompt', None)
        else:
            query = data[0]
            system_prompt = input_item.get('system_prompt', None)

        content = self.make_request_content(query, system_prompt)
        request_json = self.make_request(content, infer_cfg)
        response = self.send_request(request_json)
        return response

    def make_request_content(self, query: str, system_prompt: Optional[str] = None) -> list:
        """
        Make request content for OpenAI API.
        """
        messages = []
        if system_prompt:
            messages.append({'role': 'system', 'content': system_prompt})

        messages.append({'role': 'user', 'content': query})

        return messages

    def make_request(self, content: list, infer_cfg: dict = {}) -> dict:
        """Make request to remote API."""
        # Format request JSON according to OpenAI API format
        from evalscope.config import DEFAULT_GENERATION_CONFIG
        if infer_cfg == DEFAULT_GENERATION_CONFIG:
            infer_cfg = {
                'max_tokens': 2048,
                'temperature': 0.0,
            }

        request_json = {'model': self.model_id, 'messages': content, **infer_cfg}
        logger.debug(f'Request to remote API: {request_json}')
        return request_json

    def send_request(self, request_json: dict) -> dict:
        try:
            response: ChatCompletion = self.client.chat.completions.create(**request_json, timeout=self.timeout)
            return response.model_dump(exclude_unset=True)
        except Exception as e:
            logger.error(f'Error when calling OpenAI API: {str(e)}')
            raise
