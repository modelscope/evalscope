import requests
import time
from typing import Optional, Union

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

    def predict(self, inputs: Union[str, dict, list], infer_cfg: dict = None) -> dict:
        """
        Model prediction func.

        Args:
            inputs (Union[str, dict, list]): The input data.
            infer_cfg (dict): Inference configuration.

        Returns:
            res (dict): The model prediction results.
        """
        infer_cfg = infer_cfg or {}

        # Process inputs
        if isinstance(inputs, str):
            query = inputs
            system_prompt = None
        elif isinstance(inputs, dict):
            # TODO: to be supported for continuation list like truthful_qa
            query = inputs['data'][0]
            system_prompt = inputs.get('system_prompt', None)
        elif isinstance(inputs, list):
            query = '\n'.join(inputs)
            system_prompt = None
        else:
            raise TypeError(f'Unsupported inputs type: {type(inputs)}')

        content = self.make_request_content(query, system_prompt)
        request_json = self.make_request(content, infer_cfg)
        response = self.send_request(request_json)
        return response

    def make_request_content(self, query: str, system_prompt: Optional[str] = None) -> dict:
        """
        Make request content for API.
        """
        if system_prompt is not None:
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
        request_json = {
            **content, 'model': self.model_id,
            'max_tokens': infer_cfg.get('max_tokens', 2048),
            'temperature': infer_cfg.get('temperature', 0.0),
            'top_p': infer_cfg.get('top_p', 1.0),
            'n': infer_cfg.get('num_return_sequences', 1),
            'stop': infer_cfg.get('stop', None)
        }
        if self.seed is not None:
            request_json['seed'] = self.seed
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
