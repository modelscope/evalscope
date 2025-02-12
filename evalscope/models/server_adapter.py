import requests
import time
from typing import List, Optional, Union

from evalscope.models.base_adapter import BaseModelAdapter
from evalscope.utils.chat_service import ChatMessage
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
        self.api_url = api_url
        self.model_id = model_id
        self.api_key = api_key
        self.seed = kwargs.get('seed', None)
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

    def make_request_content(self, query: str, system_prompt: Optional[str] = None) -> dict:
        """
        Make request content for API.
        """
        if system_prompt:
            messages = [
                ChatMessage(role='system', content=system_prompt).model_dump(exclude_unset=True),
                ChatMessage(role='user', content=query).model_dump(exclude_unset=True)
            ]
        else:
            messages = [ChatMessage(role='user', content=query).model_dump(exclude_unset=True)]
        return {'messages': messages}

    def make_request(self, content: dict, infer_cfg: dict = {}) -> dict:
        """Make request to remote API."""
        # Format request JSON according to OpenAI API format
        from evalscope.config import DEFAULT_GENERATION_CONFIG
        if infer_cfg == DEFAULT_GENERATION_CONFIG:
            infer_cfg = {
                'max_tokens': 2048,
                'temperature': 0.0,
            }

        request_json = {'model': self.model_id, **content, **infer_cfg}
        logger.debug(f'Request to remote API: {request_json}')
        return request_json

    def send_request(self, request_json: dict, max_retries: int = 3) -> dict:
        for attempt in range(max_retries):
            response = requests.post(
                self.api_url, json=request_json, headers={'Authorization': f'Bearer {self.api_key}'})
            if response.status_code == 200:
                response_data = response.json()
                return response_data
            logger.warning(f'Failed to request to remote API: {response.status_code} {response.text}')
            if attempt < max_retries - 1:
                time.sleep(5)  # Sleep for 5 seconds before retrying
            else:
                raise RuntimeError(f'Failed to request to remote API after {max_retries} attempts: '
                                   f'{response.status_code} {response.text}')
