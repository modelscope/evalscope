"""OpenAI Embedding API plugin for evalscope perf."""

import json
import sys
import time
import traceback
from typing import Any, Dict, List, Tuple, Union

from evalscope.perf.arguments import Arguments
from evalscope.perf.plugin.api.base import ApiPluginBase
from evalscope.perf.plugin.registry import register_api
from evalscope.perf.utils.benchmark_util import BenchmarkData
from evalscope.utils.logger import get_logger

logger = get_logger()


@register_api(['openai_embedding', 'embedding'])
class OpenaiEmbeddingPlugin(ApiPluginBase):
    """OpenAI Embedding API plugin.

    This plugin builds requests compatible with OpenAI's embedding API format:
    POST /v1/embeddings
    {
        "input": "Your text string goes here",
        "model": "text-embedding-3-small"
    }
    """

    def __init__(self, param: Arguments):
        """Initialize the OpenaiEmbeddingPlugin.

        Args:
            param (Arguments): Configuration object containing parameters
                such as the tokenizer path and model details.
        """
        super().__init__(param=param)
        if param.tokenizer_path is not None:
            from modelscope import AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(param.tokenizer_path, trust_remote_code=True)
        else:
            self.tokenizer = None

    def build_request(self, messages: Union[List[Dict], str, List[str]], param: Arguments = None) -> Dict:
        """Build the OpenAI embedding format request.

        Args:
            messages: The input text(s) for embedding. Can be:
                - A string: single text to embed
                - A list of strings: batch of texts to embed
                - A list of dicts with 'content' key: extract text from messages
            param (Arguments): The query parameters.

        Returns:
            Dict: The request body for embedding API.
        """
        param = param or self.param
        try:
            # Extract text input from various formats
            if isinstance(messages, str):
                input_text = messages
            elif isinstance(messages, list):
                if len(messages) == 0:
                    return None
                if isinstance(messages[0], str):
                    # List of strings - batch embedding
                    input_text = messages
                elif isinstance(messages[0], dict):
                    # List of message dicts - extract content
                    contents = []
                    for msg in messages:
                        content = msg.get('content', '')
                        if isinstance(content, str):
                            contents.append(content)
                        elif isinstance(content, list):
                            # Handle multimodal content - extract text parts
                            for part in content:
                                if isinstance(part, dict) and part.get('type') == 'text':
                                    contents.append(part.get('text', ''))
                    input_text = ' '.join(contents) if len(contents) == 1 else contents
                    if isinstance(input_text, list) and len(input_text) == 1:
                        input_text = input_text[0]
                else:
                    input_text = str(messages)
            else:
                input_text = str(messages)

            # Build the embedding request
            payload = {
                'input': input_text,
                'model': param.model,
            }

            # Add optional parameters if specified
            if param.extra_args:
                payload.update(param.extra_args)

            return payload

        except Exception as e:
            logger.exception(f'Failed to build embedding request: {e}')
            return None

    def parse_responses(self, responses: List[Dict], request: str = None, **kwargs) -> Tuple[int, int]:
        """Parse embedding responses and return token counts.

        For embedding API, there's typically no "completion_tokens" since we're
        not generating text. We return (prompt_tokens, 0) or estimate from input.

        Args:
            responses: List of response dicts from the API.
            request: The original request JSON string.

        Returns:
            Tuple[int, int]: (prompt_tokens, completion_tokens)
        """
        try:
            # Embedding API typically returns usage in the response
            last_response = responses[-1] if responses else {}

            if 'usage' in last_response and last_response['usage']:
                usage = last_response['usage']
                prompt_tokens = usage.get('prompt_tokens', 0) or usage.get('total_tokens', 0)
                # Embedding doesn't have completion tokens
                return prompt_tokens, 0

            # Fallback: estimate tokens from request
            if self.tokenizer and request:
                try:
                    req_data = json.loads(request)
                    input_text = req_data.get('input', '')
                    if isinstance(input_text, list):
                        total_tokens = sum(len(self.tokenizer.encode(t, add_special_tokens=False)) for t in input_text)
                    else:
                        total_tokens = len(self.tokenizer.encode(input_text, add_special_tokens=False))
                    return total_tokens, 0
                except Exception as e:
                    logger.warning(f'Failed to estimate tokens: {e}')

            return 0, 0

        except Exception as e:
            logger.error(f'Failed to parse embedding response: {e}. Response: {responses}')
            return 0, 0

    async def process_request(self, client_session, url: str, headers: Dict, body: Dict) -> BenchmarkData:
        """Process the embedding HTTP request.

        Embedding requests are always non-streaming, so we use a simplified handler.

        Args:
            client_session: The aiohttp client session
            url: The request URL
            headers: The request headers
            body: The request body

        Returns:
            BenchmarkData: The benchmark data including response and timing info.
        """

        headers = {'Content-Type': 'application/json', **headers}
        data = json.dumps(body, ensure_ascii=False)

        output = BenchmarkData()
        st = time.perf_counter()
        output.start_time = st
        output.request = data

        try:
            async with client_session.post(url=url, data=data, headers=headers) as response:
                timestamp = time.perf_counter()
                output.completed_time = timestamp
                output.query_latency = timestamp - st
                output.first_chunk_latency = output.query_latency

                if response.status == 200:
                    try:
                        payload = await response.json()
                    except Exception:
                        payload = await response.text()

                    if isinstance(payload, dict):
                        # Extract embedding info
                        if 'data' in payload:
                            # Count dimensions for logging
                            embeddings = payload.get('data', [])
                            if embeddings:
                                embedding_dim = len(embeddings[0].get('embedding', []))
                                output.generated_text = f'embedding_dim={embedding_dim}, count={len(embeddings)}'

                        if usage := payload.get('usage'):
                            output.prompt_tokens = usage.get('prompt_tokens') or usage.get('total_tokens', 0)
                            output.completion_tokens = 0

                        output.response_messages.append(payload)
                    else:
                        output.generated_text = str(payload)

                    output.success = True
                else:
                    try:
                        err_payload = await response.json()
                        output.error = json.dumps(err_payload, ensure_ascii=False)
                    except Exception:
                        try:
                            output.error = await response.text()
                        except Exception:
                            output.error = response.reason or ''
                    output.success = False

        except Exception:
            output.success = False
            exc_info = sys.exc_info()
            output.error = ''.join(traceback.format_exception(*exc_info))
            logger.error(output.error)

        return output
