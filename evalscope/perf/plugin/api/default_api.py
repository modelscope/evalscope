import aiohttp
import codecs
import json
import sys
import time
import traceback
from typing import Any, Dict

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
        # Keep decoder state across chunks to handle split multibyte sequences
        self.decoder = codecs.getincrementaldecoder('utf-8')()

    def add_chunk(self, chunk_bytes: bytes) -> list[str]:
        """Add a chunk of bytes to the buffer and return any complete
        messages."""
        # Use incremental decoding so incomplete multibyte sequences don't error
        try:
            chunk_str = self.decoder.decode(chunk_bytes, final=False)
        except UnicodeDecodeError:
            # Bad bytes: drop them and reset decoder state to avoid corruption
            self.decoder.reset()
            chunk_str = chunk_bytes.decode('utf-8', errors='ignore')
        # Normalize CRLF (common in SSE implementations) to LF so downstream
        # splitting on "\n\n" works consistently.
        self.buffer += chunk_str.replace('\r\n', '\n')

        messages = []

        # Split by double newlines (SSE message separator)
        while '\n\n' in self.buffer:
            message, self.buffer = self.buffer.split('\n\n', 1)
            message = message.strip()
            if message:
                messages.append(message)

        # if self.buffer is not empty, check if it is a complete message
        # by removing data: prefix and check if it is a valid JSON
        if self.buffer.startswith('data:'):
            message_content = self.buffer.removeprefix('data:').strip()
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

        Returns:
            BenchmarkData: Aggregated benchmarking data for the request/response.
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
                content_type = response.headers.get('Content-Type', '')
                if response.status == 200:
                    # Handle streaming responses (SSE)
                    if 'text/event-stream' in content_type:
                        handler = StreamedResponseHandler()
                        async for chunk_bytes in response.content.iter_any():

                            if not chunk_bytes:
                                continue

                            messages = handler.add_chunk(chunk_bytes)
                            for message in messages:
                                # NOTE: SSE comments (often used as pings) start with
                                # a colon. These are not JSON data payload and should
                                # be skipped.
                                if message.startswith(':'):
                                    continue

                                chunk = message.removeprefix('data:').strip()

                                if chunk != '[DONE]':
                                    timestamp = time.perf_counter()
                                    data = json.loads(chunk)

                                    if choices := data.get('choices'):
                                        if data.get('object') == 'text_completion':
                                            content = choices[0].get('text')
                                        else:
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

                    # Handle non-stream JSON responses
                    elif 'application/json' in content_type or 'application/' in content_type:
                        payload: Any
                        try:
                            payload = await response.json()
                        except Exception:
                            # Fallback to text if JSON parsing fails
                            payload = await response.text()

                        timestamp = time.perf_counter()
                        output.completed_time = timestamp
                        output.query_latency = timestamp - st
                        # For non-stream, first chunk equals full latency
                        output.first_chunk_latency = output.query_latency

                        if isinstance(payload, dict):
                            # Extract generated text from choices
                            text = ''
                            if choices := payload.get('choices'):
                                first = choices[0] if choices else {}
                                # Chat Completions format
                                msg = first.get('message') or {}
                                if isinstance(msg, dict) and msg.get('content') is not None:
                                    text = msg.get('content') or ''
                                else:
                                    # Legacy Completions format
                                    text = first.get('text') or ''
                            generated_text = text

                            # Extract usage if provided
                            if usage := payload.get('usage'):
                                output.prompt_tokens = usage.get('prompt_tokens')
                                output.completion_tokens = usage.get('completion_tokens')

                            output.response_messages.append(payload)
                        else:
                            generated_text = str(payload)

                        output.generated_text = generated_text
                        output.success = True

                    else:
                        # Unknown successful content-type: read as text
                        raw = await response.text()
                        timestamp = time.perf_counter()
                        output.completed_time = timestamp
                        output.query_latency = timestamp - st
                        output.first_chunk_latency = output.query_latency
                        output.generated_text = raw
                        output.response_messages.append(raw)
                        output.success = True
                else:
                    # Try to parse structured error, fallback to reason/text
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
