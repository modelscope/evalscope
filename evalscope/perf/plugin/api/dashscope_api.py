import json
import os
from typing import Any, Dict, Iterator, List

from evalscope.perf.arguments import Arguments
from evalscope.perf.plugin.api.base import ApiPluginBase
from evalscope.perf.plugin.registry import register_api
from evalscope.utils.logger import get_logger

logger = get_logger()


@register_api('dashscope')
class DashScopeApiPlugin(ApiPluginBase):

    def __init__(self, mode_path: str):
        """Init the plugin

        Args:
            mode_path (str): The model path, we use the tokenizer
                weight in the model to calculate the number of the
                input and output tokens.
        """
        super().__init__(model_path=mode_path)

    def build_request(self, messages: List[Dict], param: Arguments) -> Dict:
        """Build the openai format request based on prompt, dataset

        Args:
            messages (List[Dict]): The basic message to generator query.
            param (QueryParameters): The query parameters.

        Raises:
            Exception: NotImplemented

        Returns:
            Dict: The request body. None if prompt format is error.
        """
        try:
            if param.query_template is not None:
                if param.query_template.startswith('@'):
                    file_path = param.query_template[1:]
                    if os.path.exists(file_path):
                        with open(file_path, 'r') as file:
                            query = json.load(file)
                    else:
                        raise FileNotFoundError(f'{file_path}')
                else:
                    query = json.loads(param.query_template)

                query['input']['messages'] = messages  # replace template content with message.
                return self.__compose_query_from_parameter(query, param)
            else:
                query = {'messages': messages}
                return self.__compose_query_from_parameter(query, param)
        except Exception as e:
            logger.exception(e)
            return None

    def __compose_query_from_parameter(self, payload: Dict, param: Arguments):
        payload['model'] = param.model
        if 'parameters' not in payload:
            payload['parameters'] = {}
        if param.max_tokens is not None:
            payload['parameters']['max_tokens'] = param.max_tokens
        if param.frequency_penalty is not None:
            payload['parameters']['frequency_penalty'] = param.frequency_penalty
        if param.logprobs is not None:
            payload['parameters']['logprobs'] = param.logprobs
        if param.n_choices is not None:
            payload['parameters']['n'] = param.n_choices
        if param.seed is not None:
            payload['parameters']['seed'] = param.seed
        if param.stop is not None:
            payload['parameters']['stop'] = param.stop
        if param.stream is not None and not param.stream:
            payload['parameters']['stream'] = param.stream
        if param.temperature is not None:
            payload['parameters']['temperature'] = param.temperature
        if param.top_p is not None:
            payload['parameters']['top_p'] = param.top_p
        return payload

    def parse_responses(self, responses, **kwargs) -> Dict:
        """Parser responses and return number of request and response tokens.

        Args:
            responses (List[bytes]): List of http response body, for stream output,
                there are multiple responses, for general only one.
            kwargs: (Any): The command line --parameter content.

        Returns:
            Tuple: Return number of prompt token and number of completion tokens.
        """
        last_response = responses[-1]
        js = json.loads(last_response)
        return js['usage']['input_tokens'], js['usage']['output_tokens']
