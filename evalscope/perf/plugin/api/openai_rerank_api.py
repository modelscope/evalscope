"""OpenAI/Cohere-style Rerank API plugin for evalscope perf."""
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


@register_api(['openai_rerank', 'rerank'])
class OpenaiRerankPlugin(ApiPluginBase):
    """Rerank API plugin compatible with OpenAI/Cohere/Jina style APIs.

    This plugin builds requests compatible with common rerank API formats:
    POST /v1/rerank or /rerank
    {
        "query": "What is the capital of France?",
        "documents": ["Paris is the capital of France.", "Berlin is in Germany."],
        "model": "bge-reranker-v2-m3",
        "top_n": 10
    }
    """

    def __init__(self, param: Arguments):
        """Initialize the OpenaiRerankPlugin.

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

    def build_request(self, messages: Union[List[Dict], str, Dict], param: Arguments = None) -> Dict:
        """Build the rerank format request.

        Args:
            messages: The input for reranking. Expected formats:
                - Dict with 'query' and 'documents' keys
                - List of dicts where first is query, rest are documents
                - String (treated as query with default documents)
            param (Arguments): The query parameters.

        Returns:
            Dict: The request body for rerank API.
        """
        param = param or self.param
        try:
            query = ''
            documents = []

            if isinstance(messages, dict):
                # Direct dict format with query and documents
                query = messages.get('query', '')
                documents = messages.get('documents', [])
                if isinstance(documents, str):
                    documents = [documents]
            elif isinstance(messages, list):
                if len(messages) == 0:
                    return None
                if isinstance(messages[0], dict):
                    # List of message dicts
                    # First message is query, rest are documents
                    for i, msg in enumerate(messages):
                        content = msg.get('content', '')
                        if isinstance(content, list):
                            # Handle multimodal - extract text
                            text_parts = [p.get('text', '') for p in content if p.get('type') == 'text']
                            content = ' '.join(text_parts)
                        if i == 0:
                            query = content
                        else:
                            documents.append(content)
                elif isinstance(messages[0], str):
                    # List of strings: first is query, rest are documents
                    query = messages[0]
                    documents = messages[1:] if len(messages) > 1 else ['']
            elif isinstance(messages, str):
                # Single string as query
                query = messages
                # Use a default document for testing
                documents = ['This is a test document for reranking.']

            # Build the rerank request
            payload = {
                'query': query,
                'documents': documents,
                'model': param.model,
            }

            # Add optional top_n parameter
            if param.extra_args:
                payload.update(param.extra_args)

            return payload

        except Exception as e:
            logger.exception(f'Failed to build rerank request: {e}')
            return None

    def parse_responses(self, responses: List[Dict], request: str = None, **kwargs) -> Tuple[int, int]:
        """Parse rerank responses and return token counts.

        For rerank API, we count tokens from both query and documents.

        Args:
            responses: List of response dicts from the API.
            request: The original request JSON string.

        Returns:
            Tuple[int, int]: (prompt_tokens, completion_tokens)
        """
        try:
            last_response = responses[-1] if responses else {}

            # Check for usage in response
            if 'usage' in last_response and last_response['usage']:
                usage = last_response['usage']
                prompt_tokens = usage.get('prompt_tokens', 0) or usage.get('total_tokens', 0)
                return prompt_tokens, 0

            # Some rerank APIs include token info in meta
            if 'meta' in last_response:
                meta = last_response['meta']
                if 'billed_units' in meta:
                    tokens = meta['billed_units'].get('search_units', 0)
                    return tokens, 0

            # Fallback: estimate tokens from request
            if self.tokenizer and request:
                try:
                    req_data = json.loads(request)
                    query = req_data.get('query', '')
                    documents = req_data.get('documents', [])

                    total_tokens = len(self.tokenizer.encode(query, add_special_tokens=False))
                    for doc in documents:
                        if isinstance(doc, str):
                            total_tokens += len(self.tokenizer.encode(doc, add_special_tokens=False))
                        elif isinstance(doc, dict):
                            text = doc.get('text', '')
                            total_tokens += len(self.tokenizer.encode(text, add_special_tokens=False))

                    return total_tokens, 0
                except Exception as e:
                    logger.warning(f'Failed to estimate tokens: {e}')

            return 0, 0

        except Exception as e:
            logger.error(f'Failed to parse rerank response: {e}. Response: {responses}')
            return 0, 0

    async def process_request(self, client_session, url: str, headers: Dict, body: Dict) -> BenchmarkData:
        """Process the rerank HTTP request.

        Rerank requests are always non-streaming.

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
                        # Extract rerank results info
                        results = payload.get('results', [])
                        if results:
                            # Log the top result info
                            top_result = results[0]
                            score = top_result.get('relevance_score', top_result.get('score', 0))
                            output.generated_text = f'top_score={score:.4f}, num_results={len(results)}'

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
