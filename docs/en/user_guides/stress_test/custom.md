# Custom Usage

## Custom Result Analysis

During testing, this tool saves all data into an sqlite3 database, including both requests and responses. You can analyze the test data after running your tests.

```python
import base64
import json
import pickle
import sqlite3

db_path = 'your db path'
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# Get column names
cursor.execute('PRAGMA table_info(result)')
columns = [info[1] for info in cursor.fetchall()]
print('Column names:', columns)

cursor.execute('SELECT * FROM result WHERE success=1 AND first_chunk_latency > 1')
rows = cursor.fetchall()
print(f'len(rows): {len(rows)}')

for row in rows:
    row_dict = dict(zip(columns, row))
    # Decode request
    row_dict['request'] = pickle.loads(base64.b64decode(row_dict['request']))
    # Decode response_messages
    row_dict['response_messages'] = pickle.loads(base64.b64decode(row_dict['response_messages']))
    print(
        f"request_id: {json.loads(row_dict['response_messages'][0])['id']}, first_chunk_latency: {row_dict['first_chunk_latency']}"  # noqa: E501
    )
    # If you only want to view one, you can break here
    # break
```

## Custom Request API

Currently, supported API request formats include `openai` and `dashscope`. To extend the API, you can inherit from the `ApiPluginBase` class and use the `@register_api("api_name")` decorator. You need to implement the following methods:

- **`build_request()`**  
  Build a request using `messages` as well as `model` and `query_template` in `param`. This request will be sent to the target API.

- **`process_request()`**  
  Send the request to the target API and handle the returned response (success/failure, status code, response content).

- **`parse_responses()`**  
  Parse the response and return the number of `prompt_tokens` and `completion_tokens` for inference speed calculation.

Here is a complete example code for a `custom` plugin:

```python
import json
import aiohttp
from typing import Any, AsyncGenerator, Dict, List, Tuple, Union

from evalscope.perf.arguments import Arguments
from evalscope.perf.plugin.api.base import ApiPluginBase
from evalscope.perf.plugin.registry import register_api
from evalscope.utils.logger import get_logger

logger = get_logger()


@register_api('custom')
class CustomPlugin(ApiPluginBase):
    """Template for a plugin supporting custom API implementations."""

    def __init__(self, param: Arguments):
        """Initialize the plugin, load required parameters and tokenizer.

        Args:
            param (Arguments): Configuration parameters, including:
                - tokenizer_path: Path to the tokenizer for token counting
                - model: Model name to use
                - Other request params such as max_tokens, temperature, etc.
        """
        super().__init__(param=param)
        if param.tokenizer_path is not None:
            from modelscope import AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(param.tokenizer_path)
        else:
            self.tokenizer = None

    def build_request(self, messages: Union[List[Dict], str], param: Arguments = None) -> Dict:
        """Build a custom API request body.

        Args:
            messages (Union[List[Dict], str]): Input messages, can be a list of message dicts (for chat models) or a string (for completion models).
            param (Arguments, optional): Request parameters. Defaults to self.param.

        Returns:
            Dict: Properly formatted custom API request body.
        """
        param = param or self.param
        try:
            # If no template is provided, create a default query format
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
        """Add model parameters to the request body.

        This helper adds temperature, max tokens, etc. based on what your custom API supports.

        Args:
            payload (Dict): Base request payload.
            param (Arguments): Parameters to add.

        Returns:
            Dict: Request payload with added parameters.
        """
        # Add model name
        payload['model'] = param.model
            
        # Add various parameters if provided
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

        # Add any extra parameters passed via the command line
        if param.extra_args is not None:
            payload.update(param.extra_args)
            
        return payload

    def parse_responses(self, responses: List[str], request: Any = None, **kwargs) -> Tuple[int, int]:
        """Parse responses and return token counts.

        This method extracts the number of input and output tokens from the API response.
        Different APIs may return this information in different formats, or you may need to use a tokenizer to calculate it.

        Args:
            responses (List[str]): List of API response strings.
            request (Any, optional): The original request, which may be needed for token counting.
            **kwargs: Other arguments.

        Returns:
            Tuple[int, int]: (input_tokens, output_tokens) - number of tokens in prompt and completion.
        """
        try:
            # Example 1: Try to get token count from API response
            last_response = json.loads(responses[-1])
            
            # If API provides usage info
            if 'usage' in last_response and last_response['usage']:
                input_tokens = last_response['usage'].get('prompt_tokens', 0)
                output_tokens = last_response['usage'].get('completion_tokens', 0)
                return input_tokens, output_tokens
                
            # Example 2: If no usage info, use tokenizer to count tokens
            if self.tokenizer is not None:
                input_text = ""
                output_text = ""
                
                # Extract input text from request
                if request and 'messages' in request:
                    # For chat API
                    input_text = " ".join([msg['content'] for msg in request['messages']])
                elif request and 'input_text' in request:
                    # For completion API
                    input_text = request['input_text']
                
                # Extract output text from responses
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
                
            # If neither usage info nor tokenizer is available, raise error
            raise ValueError("Cannot determine token count: no usage info in response and no tokenizer provided.")
            
        except Exception as e:
            logger.error(f"Error parsing responses: {e}")
            return 0, 0

    async def process_request(self, client_session: aiohttp.ClientSession, url: str, headers: Dict,
                              body: Dict) -> AsyncGenerator[Tuple[bool, int, str], None]:
        """Handle HTTP request and process the response.

        This method sends the request to your API and processes the response, including handling streaming responses (if supported).

        Args:
            client_session (aiohttp.ClientSession): aiohttp client session.
            url (str): API endpoint URL.
            headers (Dict): Request headers.
            body (Dict): Request body.

        Yields:
            Tuple[bool, int, str]: (is_error, status_code, response_data)
                - is_error: Whether the response indicates an error
                - status_code: HTTP status code
                - response_data: Response content
        """
        try:
            # Set content-type header
            headers = {'Content-Type': 'application/json', **headers}
            
            # Convert body to JSON
            data = json.dumps(body, ensure_ascii=False)
            
            # Send request
            async with client_session.request('POST', url=url, data=data, headers=headers) as response:
                status_code = response.status
                
                # Check if it's a stream response
                if 'text/event-stream' in response.content_type:
                    # Handle stream response
                    async for line in response.content:
                        line_str = line.decode('utf-8').strip()
                        if not line_str:
                            continue
                        
                        # Check for data prefix in server-sent events
                        if line_str.startswith('data: '):
                            data = line_str[6:]  # Remove 'data: ' prefix
                            
                            # Check for end of stream
                            if data == '[DONE]':
                                break
                                
                            try:
                                # Parse JSON data
                                parsed_data = json.loads(data)
                                yield (False, status_code, json.dumps(parsed_data))
                            except json.JSONDecodeError:
                                yield (True, status_code, f"Failed to parse JSON: {data}")
                else:
                    # Handle regular response
                    if 'application/json' in response.content_type:
                        # JSON response
                        content = await response.json()
                        yield (status_code >= 400, status_code, json.dumps(content))
                    else:
                        # Text response
                        content = await response.text()
                        yield (status_code >= 400, status_code, content)
                        
        except Exception as e:
            logger.error(f"Error in process_request: {e}")
            yield (True, 500, str(e))

if __name__ == "__main__":
    # Example usage of the custom API plugin
    from dotenv import dotenv_values
    env = dotenv_values('.env')
    
    from evalscope.perf.arguments import Arguments
    from evalscope.perf.main import run_perf_benchmark

    args = Arguments(
        model='your-model-name',
        url='https://your-api-endpoint',
        api_key='your-api-key',
        api='custom',  # Use the custom API plugin
        dataset='openqa',
        number=1,
        max_tokens=10,
        debug=True,
    )

    run_perf_benchmark(args)
```

## Custom Dataset

To customize a dataset, inherit from the `DatasetPluginBase` class and use the `@register_dataset('dataset_name')` decorator. Then implement the `build_messages` method to return a message, following the format of the [OpenAI API](https://platform.openai.com/docs/api-reference/chat/create#chat-create-messages). Specify `dataset` as your custom dataset name in the parameters to use your custom dataset.

Here is a complete example:

```python
from typing import Dict, Iterator, List

from evalscope.perf.arguments import Arguments
from evalscope.perf.plugin.datasets.base import DatasetPluginBase
from evalscope.perf.plugin.registry import register_dataset


@register_dataset('custom')
class CustomDatasetPlugin(DatasetPluginBase):
    """Read the dataset and return the prompt."""

    def __init__(self, query_parameters: Arguments):
        super().__init__(query_parameters)

    def build_messages(self) -> Iterator[List[Dict]]:
        """Build a list of messages."""
        for item in self.dataset_line_by_line(self.query_parameters.dataset_path):
            prompt = item.strip()
            if len(prompt) > self.query_parameters.min_prompt_length and len(
                    prompt) < self.query_parameters.max_prompt_length:
                if self.query_parameters.apply_chat_template:
                    yield [{'role': 'user', 'content': prompt}]
                else:
                    yield prompt


if __name__ == '__main__':
    from evalscope.perf.arguments import Arguments
    from evalscope.perf.main import run_perf_benchmark

    args = Arguments(
        model='your-model-name',
        url='https://your-api-endpoint',
        dataset_path='path/to/your/dataset.txt',  # Custom dataset path
        api_key='your-api-key',
        dataset='custom',  # Custom dataset name
    )

    run_perf_benchmark(args)
```

## Notes

1. **API Plugin Development**  
   - Ensure that you implement the `build_request`, `process_request`, and `parse_responses` methods.
   - Register your plugin using `@register_api("api_name")`.

2. **Dataset Plugin Development**  
   - Ensure that you implement the `build_messages` method.
   - Register your plugin with `@register_dataset("dataset_name")`.

3. **Debugging Tips**  
   - Use logging (`logger`) to debug plugin behavior.
   - Ensure that your API response format matches your parsing logic.

With the above examples, you can easily extend support for new API and dataset formats.