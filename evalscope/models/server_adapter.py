import requests
import time
from typing import Union

from evalscope.models.base_adapter import BaseModelAdapter
from evalscope.models.custom import CustomModel
from evalscope.models.local_model import LocalModel
from evalscope.utils.chat_service import ChatCompletionResponse


class ServerModelAdapter(BaseModelAdapter):
    """
    Server model adapter to request remote API model and generate results.
    """

    def __init__(self, model: Union[LocalModel, CustomModel], api_url: str, **kwargs):
        """
        Args:
            model: The model instance.
            api_url: The URL of the remote API model.
            **kwargs: Other args.
        """
        super().__init__(model, **kwargs)
        self.api_url = api_url

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
        elif isinstance(inputs, dict):
            # TODO: to be supported for continuation list like truthful_qa
            query = inputs['data'][0]
        elif isinstance(inputs, list):
            query = '\n'.join(inputs)
        else:
            raise TypeError(f'Unsupported inputs type: {type(inputs)}')

        # Format request JSON according to OpenAI API format
        request_json = {
            'model': self.model_id,
            'prompt': query,
            'max_tokens': infer_cfg.get('max_tokens', 2048),
            'temperature': infer_cfg.get('temperature', 1.0),
            'top_p': infer_cfg.get('top_p', 1.0),
            'n': infer_cfg.get('num_return_sequences', 1),
            'stop': infer_cfg.get('stop', None)
        }

        # Request to remote API
        response = requests.post(self.api_url, json=request_json)
        response_data = response.json()

        choices_list = [{
            'index': i,
            'message': {
                'content': choice['text'],
                'role': 'assistant'
            }
        } for i, choice in enumerate(response_data['choices'])]

        res_d = ChatCompletionResponse(
            model=self.model_id,
            choices=choices_list,
            object='chat.completion',
            created=int(time.time()),
            usage=response_data.get('usage', None)).model_dump(exclude_unset=True)

        return res_d
