import aiohttp
import asyncio
import json
import time
from http import HTTPStatus
from typing import AsyncGenerator, Dict, List, Tuple

from evalscope.perf.arguments import Arguments
from evalscope.perf.utils.local_server import ServerSentEvent
from evalscope.utils.logger import get_logger

logger = get_logger()


class AioHttpClient:

    def __init__(
        self,
        args: Arguments,
    ):
        self.url = args.url
        self.headers = {'user-agent': 'modelscope_bench', **(args.headers or {})}
        self.read_timeout = args.read_timeout
        self.connect_timeout = args.connect_timeout
        self.client = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(connect=self.connect_timeout, sock_read=self.read_timeout),
            trace_configs=[self._create_trace_config()] if args.debug else [])

    def _create_trace_config(self):
        trace_config = aiohttp.TraceConfig()
        trace_config.on_request_start.append(self.on_request_start)
        trace_config.on_request_chunk_sent.append(self.on_request_chunk_sent)
        trace_config.on_response_chunk_received.append(self.on_response_chunk_received)
        return trace_config

    async def __aenter__(self):
        pass

    async def __aexit__(self, exc_type, exc, tb):
        await self.client.close()

    async def _handle_stream(self, response: aiohttp.ClientResponse):
        is_error = False
        async for line in response.content:
            line = line.decode('utf8').rstrip('\n\r')
            sse_msg = ServerSentEvent.decode(line)
            if sse_msg:
                logger.debug(f'Response recevied: {line}')
                if sse_msg.event == 'error':
                    is_error = True
                if sse_msg.data:
                    if sse_msg.data.startswith('[DONE]'):
                        break
                    yield is_error, response.status, sse_msg.data

    async def _handle_response(self, response: aiohttp.ClientResponse) -> AsyncGenerator[Tuple[bool, int, str], None]:
        response_status = response.status
        response_content_type = response.content_type
        content_type_json = 'application/json'
        content_type_event_stream = 'text/event-stream'
        is_success = response_status == HTTPStatus.OK

        if is_success:
            # Handle successful response with 'text/event-stream' content type
            if content_type_event_stream in response_content_type:
                async for is_error, response_status, content in self._handle_stream(response):
                    yield (is_error, response_status, content)
            # Handle successful response with 'application/json' content type
            elif content_type_json in response_content_type:
                content = await response.json()
                if content.get('object') == 'error':
                    yield (True, content.get('code'), content.get('message'))  # DashScope
                else:
                    yield (False, response_status, json.dumps(content, ensure_ascii=False))
            # Handle other successful responses
            else:
                content = await response.read()
                yield (False, response_status, content)
        else:
            # Handle error response with 'application/json' content type
            if content_type_json in response_content_type:
                error = await response.json()
                yield (True, response_status, json.dumps(error, ensure_ascii=False))
            # Handle error response with 'text/event-stream' content type
            elif content_type_event_stream in response_content_type:
                async for _, _, data in self._handle_stream(response):
                    error = json.loads(data)
                    yield (True, response_status, json.dumps(error, ensure_ascii=False))
            # Handle other error responses
            else:
                msg = await response.read()
                yield (True, response_status, msg.decode('utf-8'))

    async def post(self, body):
        headers = {'Content-Type': 'application/json', **self.headers}
        try:
            data = json.dumps(body, ensure_ascii=False)  # serialize to JSON
            async with self.client.request('POST', url=self.url, data=data, headers=headers) as response:
                async for rsp in self._handle_response(response):
                    yield rsp
        except asyncio.TimeoutError:
            logger.error(
                f'TimeoutError: connect_timeout: {self.connect_timeout}, read_timeout: {self.read_timeout}. Please set longger timeout.'  # noqa: E501
            )
            yield (True, None, 'Timeout')
        except (aiohttp.ClientConnectorError, Exception) as e:
            logger.error(e)
            yield (True, None, e)

    @staticmethod
    async def on_request_start(session, context, params: aiohttp.TraceRequestStartParams):
        logger.debug(f'Starting request: <{params}>')

    @staticmethod
    async def on_request_chunk_sent(session, context, params: aiohttp.TraceRequestChunkSentParams):
        method = params.method
        url = params.url
        chunk = params.chunk.decode('utf-8')
        max_length = 100
        if len(chunk) > 2 * max_length:
            truncated_chunk = f'{chunk[:max_length]}...{chunk[-max_length:]}'
        else:
            truncated_chunk = chunk
        logger.debug(f'Request sent: <{method=},  {url=}, {truncated_chunk=}>')

    @staticmethod
    async def on_response_chunk_received(session, context, params: aiohttp.TraceResponseChunkReceivedParams):
        method = params.method
        url = params.url
        chunk = params.chunk.decode('utf-8')
        max_length = 200
        if len(chunk) > 2 * max_length:
            truncated_chunk = f'{chunk[:max_length]}...{chunk[-max_length:]}'
        else:
            truncated_chunk = chunk
        logger.debug(f'Request received: <{method=},  {url=}, {truncated_chunk=}>')


async def test_connection(args: Arguments) -> bool:
    is_error = True
    start_time = time.perf_counter()

    async def attempt_connection():
        client = AioHttpClient(args)
        async with client:
            if args.apply_chat_template:
                request = {
                    'messages': [{
                        'role': 'user',
                        'content': 'hello'
                    }],
                    'model': args.model,
                    'max_tokens': 10,
                    'stream': args.stream
                }
            else:
                request = {'prompt': 'hello', 'model': args.model, 'max_tokens': 10}
            async for is_error, state_code, response_data in client.post(request):
                return is_error, state_code, response_data

    while True:
        try:
            is_error, state_code, response_data = await asyncio.wait_for(
                attempt_connection(), timeout=args.connect_timeout)
            if not is_error:
                logger.info('Test connection successful.')
                return True
            logger.warning(f'Retrying...  <{state_code}> {response_data}')
        except Exception as e:
            logger.warning(f'Retrying... <{e}>')

        if time.perf_counter() - start_time >= args.connect_timeout:
            logger.error('Overall connection attempt timed out.')
            return False

        await asyncio.sleep(10)
