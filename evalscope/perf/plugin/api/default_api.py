import aiohttp
import json
from http import HTTPStatus
from typing import Any, AsyncGenerator, Dict, List, Tuple

from evalscope.perf.arguments import Arguments
from evalscope.perf.plugin.api.base import ApiPluginBase
from evalscope.perf.utils.local_server import ServerSentEvent
from evalscope.utils.logger import get_logger

logger = get_logger()


class DefaultApiPlugin(ApiPluginBase):
    """Default implementation of API plugin with common HTTP handling methods."""

    def __init__(self, param: Arguments):
        super().__init__(param)

    async def process_request(self, client_session: aiohttp.ClientSession, url: str, headers: Dict,
                              body: Dict) -> AsyncGenerator[Tuple[bool, int, str], None]:
        """Process the HTTP request and handle the response.

        Args:
            client_session: The aiohttp client session
            url: The request URL
            headers: The request headers
            body: The request body

        Yields:
            Tuple[bool, int, str]: (is_error, status_code, response_data)
        """
        try:
            headers = {'Content-Type': 'application/json', **headers}
            data = json.dumps(body, ensure_ascii=False)  # serialize to JSON
            async with client_session.request('POST', url=url, data=data, headers=headers) as response:
                async for result in self._handle_response(response):
                    yield result
        except Exception as e:
            logger.error(f'Error in process_request: {e}')
            yield (True, None, str(e))

    async def _handle_stream(self, response: aiohttp.ClientResponse) -> AsyncGenerator[Tuple[bool, int, str], None]:
        """Handle streaming response from server-sent events.

        Args:
            response: The aiohttp response object containing a stream

        Yields:
            Tuple[bool, int, str]: (is_error, status_code, data)
        """
        is_error = False
        async for line in response.content:
            line = line.decode('utf8').rstrip('\n\r')
            sse_msg = ServerSentEvent.decode(line)
            if sse_msg:
                logger.debug(f'Response received: {line}')
                if sse_msg.event == 'error':
                    is_error = True
                if sse_msg.data:
                    if sse_msg.data.startswith('[DONE]'):
                        break
                    yield is_error, response.status, sse_msg.data

    async def _handle_response(self, response: aiohttp.ClientResponse) -> AsyncGenerator[Tuple[bool, int, str], None]:
        """Handle the HTTP response based on content type and status.

        Args:
            response: The aiohttp response object

        Yields:
            Tuple[bool, int, str]: (is_error, status_code, response_data)
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
                yield (False, response_status, json.dumps(content, ensure_ascii=False))
            # Handle other successful responses
            else:
                content = await response.read()
                yield (False, response_status, content.decode('utf-8'))
        else:
            # error is always in JSON format
            error = await response.json()
            yield (True, response_status, json.dumps(error, ensure_ascii=False))
