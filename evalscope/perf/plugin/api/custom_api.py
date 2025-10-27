import aiohttp
import json
from typing import Any, AsyncGenerator, Dict, List, Tuple, Union

from evalscope.perf.arguments import Arguments
from evalscope.perf.plugin.api.base import ApiPluginBase
from evalscope.perf.plugin.registry import register_api
from evalscope.perf.utils.benchmark_util import BenchmarkData
from evalscope.utils.logger import get_logger

logger = get_logger()


@register_api('custom')
class CustomPlugin(ApiPluginBase):
    """Support custom API implementations.

    This class serves as a template for users to implement their own API plugins.
    By extending this class, users can connect to any LLM API with custom request
    and response formats.
    """

    def __init__(self, param: Arguments):
        """Initialize the plugin with the provided parameters.

        Args:
            param (Arguments): Configuration parameters for the plugin, including:
                - tokenizer_path: Path to the tokenizer for token counting
                - model: Name of the model to use
                - Other request parameters like max_tokens, temperature, etc.
        """
        super().__init__(param=param)
        if param.tokenizer_path is not None:
            from modelscope import AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(param.tokenizer_path)
        else:
            self.tokenizer = None

    def build_request(self, messages: Union[List[Dict], str], param: Arguments = None) -> Dict:
        """Build a custom API request body based on the input messages and parameters.

        This method formats the input messages into the expected request format
        for your custom API.

        Args:
            messages (Union[List[Dict], str]): The input messages to include in the request.
                Could be a list of message dictionaries (for chat models) or a string (for completion models).
            param (Arguments, optional): Request parameters. Defaults to self.param.

        Returns:
            Dict: A properly formatted request body for your custom API.
        """
        param = param or self.param
        try:
            # Create a default query format if no template is provided
            if isinstance(messages, str):
                query = {'input_text': messages}
            else:
                query = {'messages': messages}

            # Add model parameters to the request
            return self._add_parameters_to_request(query, param)
        except Exception as e:
            logger.exception(e)
            return None

    def _add_parameters_to_request(self, payload: Dict, param: Arguments) -> Dict:
        """Add model parameters to the request payload.

        This helper method adds various parameters like temperature, max_tokens, etc.
        to the request based on what your custom API supports.

        Args:
            payload (Dict): The base request payload.
            param (Arguments): The parameters to add.

        Returns:
            Dict: The request payload with added parameters.
        """
        # Add the model name
        payload['model'] = param.model

        # Add various parameters if they are provided
        if param.max_tokens is not None:
            payload['max_tokens'] = param.max_tokens
        if param.temperature is not None:
            payload['temperature'] = param.temperature
        if param.top_p is not None:
            payload['top_p'] = param.top_p
        if param.top_k is not None:
            payload['top_k'] = param.top_k
        if param.stream is not None:
            payload['stream'] = param.stream
            payload['stream_options'] = {'include_usage': True}

        # Add any extra arguments passed via command line
        if param.extra_args is not None:
            payload.update(param.extra_args)

        return payload

    def parse_responses(self, responses: List[Dict], request: str = None, **kwargs) -> Tuple[int, int]:
        """Parse API responses and return token counts.

        This method extracts the number of input and output tokens from the API responses.
        Different APIs may return this information in different formats, or you may need
        to calculate it using a tokenizer.

        Args:
            responses (List[Dict]): List of API response strings.
            request (str, optional): The original request, which might be needed for token calculation.
            **kwargs: Additional arguments.

        Returns:
            Tuple[int, int]: (input_tokens, output_tokens) - The number of tokens in the prompt and completion.
        """
        try:
            # Example 1: Try to get token counts from the API response
            last_response = json.loads(responses[-1])

            # If the API provides token usage information
            if 'usage' in last_response and last_response['usage']:
                input_tokens = last_response['usage'].get('prompt_tokens', 0)
                output_tokens = last_response['usage'].get('completion_tokens', 0)
                return input_tokens, output_tokens

            # Example 2: Calculate tokens using the tokenizer if no usage info is available
            if self.tokenizer is not None:
                input_text = ''
                output_text = ''

                # Extract input text from the request
                if request and 'messages' in request:
                    # For chat API
                    input_text = ' '.join([msg['content'] for msg in request['messages']])
                elif request and 'input_text' in request:
                    # For completion API
                    input_text = request['input_text']

                # Extract output text from the response
                for response in responses:
                    js = json.loads(response)
                    if 'choices' in js:
                        for choice in js['choices']:
                            if 'message' in choice and 'content' in choice['message']:
                                output_text += choice['message']['content']
                            elif 'text' in choice:
                                output_text += choice['text']

                # Count tokens
                input_tokens = len(self.tokenizer.encode(input_text))
                output_tokens = len(self.tokenizer.encode(output_text))
                return input_tokens, output_tokens

            # If no usage information and no tokenizer, raise an error
            raise ValueError(
                'Cannot determine token counts: no usage information in response and no tokenizer provided.'
            )

        except Exception as e:
            logger.error(f'Error parsing responses: {e}')
            return 0, 0

    async def process_request(
        self, client_session: aiohttp.ClientSession, url: str, headers: Dict, body: Dict
    ) -> BenchmarkData:
        """Process the HTTP request and handle the response.

        This method handles sending the request to your API and processing the response,
        including handling streaming responses if supported.

        Args:
            client_session (aiohttp.ClientSession): The aiohttp client session.
            url (str): The API endpoint URL.
            headers (Dict): The request headers.
            body (Dict): The request body.

        Returns:
            BenchmarkData: The benchmark data including response and timing info.
        """
        raise NotImplementedError(
            'The `process_request` method must be implemented in a subclass. '
            'For OpenAI-compatible APIs, consider inheriting from `DefaultApiPlugin` to reuse the default implementation.'  # noqa: E501
        )


if __name__ == '__main__':
    # Example usage of the CustomPlugin
    from dotenv import dotenv_values
    env = dotenv_values('.env')

    from evalscope.perf.arguments import Arguments
    from evalscope.perf.main import run_perf_benchmark

    args = Arguments(
        model='qwen2.5-7b-instruct',
        url='https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions',
        api_key=env.get('DASHSCOPE_API_KEY'),
        api='custom',  # Use the custom API plugin registered above
        dataset='openqa',
        number=1,
        max_tokens=10
    )

    run_perf_benchmark(args)
