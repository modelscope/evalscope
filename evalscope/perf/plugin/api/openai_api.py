import json
import os
from typing import Any, Dict, Iterator, List, Union

from evalscope.perf.arguments import Arguments
from evalscope.perf.plugin.api.base import ApiPluginBase
from evalscope.perf.plugin.registry import register_api
from evalscope.utils.logger import get_logger

logger = get_logger()


@register_api(['openai', 'local_vllm', 'local'])
class OpenaiPlugin(ApiPluginBase):
    """Base of openai interface."""

    def __init__(self, mode_path: str):
        """Init the plugin

        Args:
            mode_path (str): The model path, we use the tokenizer
                weight in the model to calculate the number of the
                input and output tokens.
        """
        super().__init__(model_path=mode_path)
        if mode_path is not None:
            from modelscope import AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(mode_path)
        else:
            self.tokenizer = None

    def build_request(self, messages: Union[List[Dict], str], param: Arguments) -> Dict:
        """Build the openai format request based on prompt, dataset

        Args:
            message (List[Dict] | str): The basic message to generator query.
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

                if 'stream' in query.keys():
                    param.stream = query['stream']
                # replace template messages with input messages.
                query['messages'] = messages
            elif isinstance(messages, str):
                query = {'prompt': messages}
            else:
                query = {'messages': messages}
            return self.__compose_query_from_parameter(query, param)
        except Exception as e:
            logger.exception(e)
            return None

    def __compose_query_from_parameter(self, payload: Dict, param: Arguments):
        payload['model'] = param.model
        if param.max_tokens is not None:
            payload['max_tokens'] = param.max_tokens
        if param.min_tokens is not None:
            payload['min_tokens'] = param.min_tokens
        if param.frequency_penalty is not None:
            payload['frequency_penalty'] = param.frequency_penalty
        if param.logprobs is not None:
            payload['logprobs'] = param.logprobs
        if param.n_choices is not None:
            payload['n'] = param.n_choices
        if param.seed is not None:
            payload['seed'] = param.seed
        if param.stop is not None:
            payload['stop'] = param.stop
        if param.stream is not None and param.stream:
            payload['stream'] = param.stream
            payload['stream_options'] = {'include_usage': True}
        if param.stop_token_ids is not None:
            payload['stop_token_ids'] = param.stop_token_ids
        if param.temperature is not None:
            payload['temperature'] = param.temperature
        if param.top_p is not None:
            payload['top_p'] = param.top_p
        if param.top_k is not None:
            payload['top_k'] = param.top_k
        if param.extra_args is not None:
            payload.update(param.extra_args)
        return payload

    def parse_responses(self, responses, request: Any = None, **kwargs) -> tuple[int, int]:
        """Parser responses and return number of request and response tokens.
        Only one response for non-stream, multiple responses for stream.
        """

        # when stream, the last response is the full usage
        # when non-stream, the last response is the first response
        last_response_js = json.loads(responses[-1])
        if 'usage' in last_response_js and last_response_js['usage']:
            input_tokens = last_response_js['usage']['prompt_tokens']
            output_tokens = last_response_js['usage']['completion_tokens']
            return input_tokens, output_tokens

        # no usage information in the response, parse the response to get the tokens
        delta_contents = {}
        for response in responses:
            js = json.loads(response)
            if 'object' in js:
                self.__process_response_object(js, delta_contents)
            else:
                self.__process_no_object(js, delta_contents)

        input_tokens, output_tokens = self.__calculate_tokens_from_content(request, delta_contents)
        return input_tokens, output_tokens

    def __process_response_object(self, js, delta_contents):
        if js['object'] == 'chat.completion':
            for choice in js['choices']:
                delta_contents[choice['index']] = [choice['message']['content']]
        elif js['object'] == 'text_completion':
            for choice in js['choices']:
                delta_contents[choice['index']] = [choice['text']]
        elif js['object'] == 'chat.completion.chunk':
            for choice in js.get('choices', []):
                if 'delta' in choice and 'index' in choice:
                    delta = choice['delta']
                    idx = choice['index']
                    if 'content' in delta:
                        delta_content = delta['content']
                        delta_contents.setdefault(idx, []).append(delta_content)

    def __process_no_object(self, js, delta_contents):
        #  assume the response is a single choice
        for choice in js['choices']:
            if 'delta' in choice:
                delta = choice['delta']
                idx = choice['index']
                if 'content' in delta:
                    delta_content = delta['content']
                    delta_contents.setdefault(idx, []).append(delta_content)
            else:
                delta_contents[choice['index']] = [choice['message']['content']]

    def __calculate_tokens_from_content(self, request, delta_contents):
        input_tokens = output_tokens = 0
        if self.tokenizer is not None:
            for idx, choice_contents in delta_contents.items():
                full_response_content = ''.join(choice_contents)
                input_tokens += len(self.tokenizer.encode(request['messages'][0]['content']))
                output_tokens += len(self.tokenizer.encode(full_response_content))
        else:
            raise ValueError('Error: Unable to retrieve usage information\n\n'
                             'This error occurs when:\n'
                             '1. The API response does not contain usage data, AND\n'
                             '2. No tokenizer has been specified or found.\n\n'
                             'To resolve this issue, do ONE of the following:\n'
                             "a) Ensure that the API you're using supports and returns usage information, OR\n"
                             'b) Specify a tokenizer using the `--tokenizer-path` parameter.\n\n'
                             'If you continue to experience issues, '
                             'please open an issue on our GitHub repository https://github.com/modelscope/evalscope .')
        return input_tokens, output_tokens
