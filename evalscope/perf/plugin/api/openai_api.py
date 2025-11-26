import json
import math
import os
from collections import defaultdict
from typing import Any, Dict, List, Tuple, Union

from evalscope.perf.arguments import Arguments
from evalscope.perf.plugin.api.default_api import DefaultApiPlugin
from evalscope.perf.plugin.registry import register_api
from evalscope.utils.io_utils import base64_to_PIL
from evalscope.utils.logger import get_logger

logger = get_logger()


@register_api(['openai', 'local_vllm', 'local'])
class OpenaiPlugin(DefaultApiPlugin):
    """Base of openai interface."""

    def __init__(self, param: Arguments):
        """Initialize the OpenaiPlugin.

        Args:
            param (Arguments): Configuration object containing parameters
                such as the tokenizer path and model details. If a tokenizer
                path is provided, it is used to initialize the tokenizer.
        """
        super().__init__(param=param)
        if param.tokenizer_path is not None:
            from modelscope import AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(param.tokenizer_path)
        else:
            self.tokenizer = None

    def build_request(self, messages: Union[List[Dict], str], param: Arguments = None) -> Dict:
        """Build the openai format request based on prompt, dataset

        Args:
            message (List[Dict] | str): The basic message to generator query.
            param (QueryParameters): The query parameters.

        Raises:
            Exception: NotImplemented

        Returns:
            Dict: The request body. None if prompt format is error.
        """
        param = param or self.param
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
        if param.repetition_penalty is not None:
            payload['repetition_penalty'] = param.repetition_penalty
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

    def parse_responses(self, responses: List[Dict], request: str = None, **kwargs) -> tuple[int, int]:
        """Parser responses and return number of request and response tokens.
        Only one response for non-stream, multiple responses for stream.
        """

        try:
            # when stream, the last response is the full usage
            # when non-stream, the last response is the first response
            last_response_js = responses[-1]
            if 'usage' in last_response_js and last_response_js['usage']:
                input_tokens = last_response_js['usage']['prompt_tokens']
                output_tokens = last_response_js['usage']['completion_tokens']
                return input_tokens, output_tokens
        except Exception as e:
            logger.error(f'Failed to parse usage from response: {e}. Response: {responses}')
            return 0, 0

        # no usage information in the response, parse the response to get the tokens
        delta_contents = defaultdict(list)
        for response in responses:
            if 'object' in response:
                self.__process_response_object(response, delta_contents)
            else:
                self.__process_no_object(response, delta_contents)

        input_tokens, output_tokens = self.__calculate_tokens_from_content(request, delta_contents)
        return input_tokens, output_tokens

    def __process_response_object(self, response, delta_contents):
        if not response.get('choices'):
            return
        if response['object'] == 'chat.completion':
            for choice in response['choices']:
                delta_contents[choice['index']] = [choice['message']['content']]
        elif response['object'] == 'text_completion':
            for choice in response['choices']:
                if 'text' in choice and 'index' in choice:
                    delta_contents[choice['index']].append(choice['text'])
        elif response['object'] == 'chat.completion.chunk':
            for choice in response['choices']:
                if 'delta' in choice and 'index' in choice:
                    delta = choice['delta']
                    idx = choice['index']
                    if 'content' in delta:
                        delta_contents[idx].append(delta['content'])

    def __process_no_object(self, response, delta_contents):
        #  assume the response is a single choice
        if not response.get('choices'):
            return
        for choice in response['choices']:
            if 'delta' in choice:
                delta = choice['delta']
                idx = choice['index']
                if 'content' in delta:
                    delta_contents[idx].append(delta['content'])
            else:
                delta_contents[choice['index']] = [choice['message']['content']]

    def __calculate_tokens_from_content(self, request, content):
        input_tokens = output_tokens = 0
        if self.tokenizer is not None:
            # Calculate input tokens
            input_tokens += self._count_input_tokens(request)
            for idx, choice_contents in content.items():
                full_response_content = ''.join(choice_contents)
                # Calculate output tokens
                output_tokens += self._count_output_tokens(full_response_content)
        else:
            raise ValueError(
                'Error: Unable to retrieve usage information\n\n'
                'This error occurs when:\n'
                '1. The API response does not contain usage data, AND\n'
                '2. No tokenizer has been specified or found.\n\n'
                'To resolve this issue, do ONE of the following:\n'
                "a) Ensure that the API you're using supports and returns usage information, OR\n"
                'b) Specify a tokenizer using the `--tokenizer-path` parameter.\n\n'
                'If you continue to experience issues, '
                'please open an issue on our GitHub repository https://github.com/modelscope/evalscope .'
            )
        return input_tokens, output_tokens

    def _count_input_tokens(self, request_str: str) -> int:
        """Count the number of input tokens in the request.

        This method handles different types of requests and calculates tokens for:
        - Text content in messages or prompts
        - Images in multimodal messages (converted to patch tokens)

        Args:
            request_str (str): The request json str containing either 'messages' for chat
                          completion or 'prompt' for text completion.

        Returns:
            int: The total number of input tokens including text and image tokens.
        """
        input_tokens = 0
        request = json.loads(request_str)
        if 'messages' in request:
            input_content = self.tokenizer.apply_chat_template(
                request['messages'], tokenize=True, add_generation_prompt=True
            )
            input_tokens += len(input_content)
            # handle image tokens if any
            for message in request['messages']:
                content = message.get('content', '')
                if isinstance(content, str):
                    continue
                for cont in content:
                    if cont['type'] == 'image_url':
                        try:
                            # assuming image_url is base64 string
                            image_base64 = cont['image_url']['url']
                            image = base64_to_PIL(image_base64)
                            # Use math.ceil for more accurate token count when image dimensions
                            # aren't perfectly divisible by patch size
                            n_patches = (
                                math.ceil(image.height / self.param.image_patch_size)
                                * math.ceil(image.width / self.param.image_patch_size)
                            )
                            input_tokens += n_patches
                        except Exception as e:
                            logger.warning(f'Failed to process image for token counting: {e}')
                            # Continue processing other content without failing
        elif 'prompt' in request:
            input_tokens += len(self.tokenizer.encode(request['prompt'], add_special_tokens=False))
        return input_tokens

    def _count_output_tokens(self, response: str) -> int:
        """Count the number of output tokens in the response. Only string response is supported.

        Args:
            response (str): The API response text.

        Returns:
            int: The number of output tokens.
        """
        return len(self.tokenizer.encode(response, add_special_tokens=False))
