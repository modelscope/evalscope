# Custom Usage

## Custom Result Analysis
During testing, this tool saves all data, including requests and responses, to an SQLite3 database. After testing, you can analyze the test data.

```python
import base64
import json
import pickle
import sqlite3

db_path = 'your db path'
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# Retrieve column names
cursor.execute('PRAGMA table_info(result)')
columns = [info[1] for info in cursor.fetchall()]
print('Columns:', columns)

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
    # If you only want to view one, you can break
    # break
```

## Custom API Requests
Currently, `openai` and `dashscope` are built-in and supported. To extend an API, inherit from `ApiPluginBase` or `DefaultApiPlugin`, and register the plugin using `@register_api("api_name")`. You must implement the following methods:

- build_request(messages, param) -> Dict  
  Construct the request body from input based on parameters such as `param.model`, `param.max_tokens`, `param.temperature`, etc.

- parse_responses(responses: List[Dict], request: str | None = None) -> Tuple[int, int]  
  Parse the responses and return `(prompt_tokens, completion_tokens)`. If the API doesn’t provide usage data, you can use a tokenizer for estimation.

- process_request(...) -> BenchmarkData  
  Send the request, and gather the responses and latency data. If your custom API is compatible with OpenAI (using JSON + SSE), inheriting from `DefaultApiPlugin` is recommended. You can reuse its HTTP and streaming functionalities and only need to implement `build_request` and `parse_responses`.

Example: Minimum implementation by inheriting `DefaultApiPlugin` (recommended)

```python
# This is an example for documentation purposes; the actual file can be found at evalscope/perf/plugin/api/custom_api.py
import json
from typing import Any, Dict, List, Tuple, Union

from evalscope.perf.arguments import Arguments
from evalscope.perf.plugin.api.default_api import DefaultApiPlugin
from evalscope.perf.plugin.registry import register_api
from evalscope.utils.logger import get_logger

logger = get_logger()


@register_api('custom')
class CustomPlugin(DefaultApiPlugin):
    """Custom API plugin (recommended to inherit from DefaultApiPlugin for OpenAI-compatible APIs)."""

    def __init__(self, param: Arguments):
        super().__init__(param)
        # Optional: Used for token estimation when the API doesn’t return the usage
        if param.tokenizer_path is not None:
            from modelscope import AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(param.tokenizer_path)
        else:
            self.tokenizer = None

    def build_request(self, messages: Union[List[Dict], str], param: Arguments = None) -> Dict:
        """Construct the request body for the custom API from input messages/strings."""
        param = param or self.param
        try:
            if isinstance(messages, str):
                payload = {'input_text': messages}
            else:
                payload = {'messages': messages}

            # Add common inference parameters
            payload['model'] = param.model
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
            if param.extra_args:
                payload.update(param.extra_args)

            return payload
        except Exception as e:
            logger.exception(e)
            return {}

    def parse_responses(self, responses: List[Dict], request: str = None, **kwargs: Any) -> Tuple[int, int]:
        """Extract token counts from the response list; estimate tokens if usage is not returned."""
        try:
            last = responses[-1] if responses else {}
            if isinstance(last, dict) and last.get('usage'):
                usage = last['usage'] or {}
                return usage.get('prompt_tokens', 0), usage.get('completion_tokens', 0)

            # Fallback: Estimate tokens using the tokenizer
            if self.tokenizer is not None:
                prompt_text = ''
                if request:
                    try:
                        req_js = json.loads(request)
                        if isinstance(req_js, dict):
                            if 'messages' in req_js:
                                prompt_text = ' '.join(m.get('content', '') for m in req_js.get('messages', []))
                            elif 'input_text' in req_js:
                                prompt_text = req_js.get('input_text') or ''
                    except Exception:
                        pass

                completion_text = ''
                for resp in responses:
                    if not isinstance(resp, dict):
                        continue
                    for choice in resp.get('choices', []) or []:
                        msg = choice.get('message') or {}
                        if isinstance(msg, dict) and msg.get('content'):
                            completion_text += msg.get('content') or ''
                        else:
                            completion_text += choice.get('text') or ''

                return len(self.tokenizer.encode(prompt_text)), len(self.tokenizer.encode(completion_text))

            return 0, 0
        except Exception as e:
            logger.error(f'Error parsing response: {e}')
            return 0, 0
```

Usage example:

```python
from dotenv import dotenv_values
from evalscope.perf.arguments import Arguments
from evalscope.perf.main import run_perf_benchmark

env = dotenv_values('.env')

args = Arguments(
    model='your-model',
    url='https://your-endpoint',
    api_key=env.get('YOUR_API_KEY'),
    api='custom',     # Use the above registered plugin
    dataset='openqa',
    number=1,
    max_tokens=16,
    stream=True,      # If streaming is supported
    debug=True,
)

run_perf_benchmark(args)
```

If your API is not compatible with OpenAI streaming protocol, you need to implement `process_request(...) -> BenchmarkData` in the custom plugin (refer to the implementation in `evalscope/perf/plugin/api/default_api.py`).

## Custom Dataset

To create a custom dataset, inherit from the `DatasetPluginBase` class, use the `@register_dataset('dataset_name')` decorator, and implement the `build_messages` method to return a message in the format outlined in [OpenAI API](https://platform.openai.com/docs/api-reference/chat/create#chat-create-messages). Specify `dataset` as the custom dataset name in the arguments to use the custom dataset.

Below is a complete example:

```python
from typing import Dict, Iterator, List

from evalscope.perf.arguments import Arguments
from evalscope.perf.plugin.datasets.base import DatasetPluginBase
from evalscope.perf.plugin.registry import register_dataset


@register_dataset('custom')
class CustomDatasetPlugin(DatasetPluginBase):
    """Reads the dataset and returns prompts."""

    def __init__(self, query_parameters: Arguments):
        super().__init__(query_parameters)

    def build_messages(self) -> Iterator[List[Dict]]:
        """Construct the message list."""
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

1. API Plugin Development  
   - You must implement `build_request` and `parse_responses` and provide `process_request(...) -> BenchmarkData` (or inherit from `DefaultApiPlugin` to reuse the default implementation).
   - Use `@register_api("api_name")` to register the plugin.
   - Prefer using `DefaultApiPlugin` to reuse common logic for HTTP, SSE, and usage collection.

2. Dataset Plugin Development  
   - Implement `build_messages` and register it with `@register_dataset("dataset_name")`.

3. Debugging Tips  
   - Use `logger` to output key information.
   - Ensure the response structure matches the parsing logic. If necessary, print the raw response for troubleshooting.