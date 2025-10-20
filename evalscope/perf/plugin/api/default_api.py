import aiohttp
import json
import sys
import time
import traceback
from http import HTTPStatus
from typing import Any, AsyncGenerator, Dict, List, Tuple

from evalscope.perf.arguments import Arguments
from evalscope.perf.plugin.api.base import ApiPluginBase
from evalscope.perf.utils.benchmark_util import BenchmarkData
from evalscope.utils.logger import get_logger

logger = get_logger()


class StreamedResponseHandler:
    """Handles streaming HTTP responses by accumulating chunks until complete
    messages are available."""

    def __init__(self):
        self.buffer = ''

    def add_chunk(self, chunk_bytes: bytes) -> list[str]:
        """Add a chunk of bytes to the buffer and return any complete
        messages."""
        chunk_str = chunk_bytes.decode('utf-8')
        self.buffer += chunk_str

        messages = []

        # Split by double newlines (SSE message separator)
        while '\n\n' in self.buffer:
            message, self.buffer = self.buffer.split('\n\n', 1)
            message = message.strip()
            if message:
                messages.append(message)

        # if self.buffer is not empty, check if it is a complete message
        # by removing data: prefix and check if it is a valid JSON
        if self.buffer.startswith('data: '):
            message_content = self.buffer.removeprefix('data: ').strip()
            if message_content == '[DONE]':
                messages.append(self.buffer.strip())
                self.buffer = ''
            elif message_content:
                try:
                    json.loads(message_content)
                    messages.append(self.buffer.strip())
                    self.buffer = ''
                except json.JSONDecodeError:
                    # Incomplete JSON, wait for more chunks.
                    pass

        return messages


class DefaultApiPlugin(ApiPluginBase):
    """Default implementation of API plugin with common HTTP handling methods."""

    def __init__(self, param: Arguments):
        super().__init__(param)

    async def process_request(
        self, client_session: aiohttp.ClientSession, url: str, headers: Dict, body: Dict
    ) -> BenchmarkData:
        """Process the HTTP request and handle the response.

        Args:
            client_session: The aiohttp client session
            url: The request URL
            headers: The request headers
            body: The request body

        Yields:
            Tuple[bool, int, Any]: (is_error, status_code, response_data)
        """
        headers = {'Content-Type': 'application/json', **headers}
        data = json.dumps(body, ensure_ascii=False)  # serialize to JSON

        output = BenchmarkData()
        ttft = 0.0
        generated_text = ''
        st = time.perf_counter()
        output.start_time = st
        output.request = data
        most_recent_timestamp = st
        try:
            async with client_session.post(url=url, data=data, headers=headers) as response:
                if response.status == 200:
                    handler = StreamedResponseHandler()
                    async for chunk_bytes in response.content.iter_any():
                        chunk_bytes = chunk_bytes.strip()
                        if not chunk_bytes:
                            continue

                        messages = handler.add_chunk(chunk_bytes)
                        for message in messages:
                            # NOTE: SSE comments (often used as pings) start with
                            # a colon. These are not JSON data payload and should
                            # be skipped.
                            if message.startswith(':'):
                                continue

                            chunk = message.removeprefix('data: ')

                            if chunk != '[DONE]':
                                timestamp = time.perf_counter()
                                data = json.loads(chunk)

                                if choices := data.get('choices'):
                                    content = choices[0]['delta'].get('content')
                                    # First token
                                    if ttft == 0.0:
                                        ttft = timestamp - st
                                        output.first_chunk_latency = ttft

                                    # Decoding phase
                                    else:
                                        output.inter_chunk_latency.append(timestamp - most_recent_timestamp)

                                    generated_text += content or ''
                                    output.response_messages.append(data)
                                elif usage := data.get('usage'):
                                    output.prompt_tokens = usage.get('prompt_tokens')
                                    output.completion_tokens = usage.get('completion_tokens')

                                most_recent_timestamp = timestamp

                    output.generated_text = generated_text
                    output.success = True
                    output.completed_time = most_recent_timestamp
                    output.query_latency = most_recent_timestamp - st
                else:
                    output.error = response.reason or ''
                    output.success = False
        except Exception:
            output.success = False
            exc_info = sys.exc_info()
            output.error = ''.join(traceback.format_exception(*exc_info))
            logger.error(output.error)

        return output

    async def _handle_response(self, response: aiohttp.ClientResponse) -> AsyncGenerator[Tuple[bool, int, Any], None]:
        """Handle the HTTP response based on content type and status.

        Args:
            response: The aiohttp response object

        Yields:
            Tuple[bool, int, Any]: (is_error, status_code, response_data)
        """
        response_status = response.status
        response_content_type = response.content_type
        content_type_json = 'application/json'
        content_type_stream = 'text/event-stream'
        is_success = (response_status == HTTPStatus.OK)

        if is_success:
            # Handle successful response with 'text/event-stream' content type
            if content_type_stream in response_content_type:
                async for is_error, response_status, content in self._handle_stream(response):
                    yield (is_error, response_status, content)
            # Handle successful response with 'application/json' content type
            elif content_type_json in response_content_type:
                content = await response.json()
                yield (False, response_status, content)
            # Handle other successful responses
            else:
                content = await response.read()
                yield (False, response_status, content.decode('utf-8'))
        else:
            # error is always in JSON format
            error = await response.json()
            yield (True, response_status, error)
