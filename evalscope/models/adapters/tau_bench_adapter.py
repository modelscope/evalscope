import json
import time
import uuid
from typing import Any, List, Optional, Union

from evalscope.utils.logger import get_logger
from ..register import register_model_adapter
from .server_adapter import ServerModelAdapter

logger = get_logger()


@register_model_adapter(name='tau_bench')
class TauBenchAdapter(ServerModelAdapter):
    """
    TauBench model adapter to request remote API model and generate results for TauBench evaluation.
    Support multi-turn and single-turn function calling tasks.
    """

    def __init__(self, api_url: str, model_id: str, api_key: str = 'EMPTY', **kwargs):
        """
        Args:
            api_url: The URL of the remote API model.
            model_id: The ID of the remote API model.
            api_key: The API key of the remote API model.
        """
        super().__init__(api_url=api_url, model_id=model_id, api_key=api_key, **kwargs)

    def predict(self, inputs: List[dict], infer_cfg: Optional[dict] = None) -> List[dict]:
        """
        Model prediction func. For multi-turn evals, we pass a list[list[message]] to the model
        where each list is a follow up turn in the conversation
        each turn is a List[List[Message]]

        Args:
            inputs (List[dict]): The input data.
            infer_cfg (dict): Inference configuration.

        Returns:
            res (List[dict]): The model prediction results.
        """
        infer_cfg = infer_cfg or {}
        results = []

        for input_item in inputs:
            # This flag decides if we pass tools to the API or try tool calling via prompting
            # Passing tools to the API means that we rely on the API to manage system prompt specifics
            # and also expect parsed tool calls in the ChatCompletionMessage object
            # This is how the is_fc_model=True benchmark is designed to work
            # On the other hand, we try to manage
            # tool calling via prompting and parse tool calls in the standard text response
            # This is how the is_fc_model=False benchmark is designed to work
            row = input_item.get('messages')
            is_fc_model = row.get('is_fc_model', False)

            if is_fc_model:
                response = self.generate_turn_with_tools(row, infer_cfg)
            else:
                response = self.generate_turn(row, infer_cfg)

            # wrap response with openai types
            res_d = {
                'choices': [{
                    'index': 0,
                    'message': {
                        'content': response,
                        'role': 'assistant'
                    }
                }],
                'created': time.time(),
                'model': self.model_id,
                'object': 'chat.completion',
                'usage': {
                    'completion_tokens': 0,
                    'prompt_tokens': 0,
                    'total_tokens': 0
                }
            }
            results.append(res_d)

        return results
