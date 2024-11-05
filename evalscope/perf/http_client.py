import logging
from http import HTTPStatus
from typing import Dict, List

import aiohttp
import json

from evalscope.perf.utils._logging import logger
from evalscope.perf.utils.server_sent_event import ServerSentEvent


class AioHttpClient:

    def __init__(
        self,
        url: str,
        conn_timeout: int = 120,
        read_timeout: int = 120,
        headers: Dict = None,
        debug: bool = False,
    ):
        self.url = url
        self.debug = debug
        self.headers = {'user-agent': 'modelscope_bench', **(headers or {})}
        self.client = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(
                total=read_timeout + conn_timeout,
                connect=conn_timeout,
                sock_read=read_timeout),
            connector=aiohttp.TCPConnector(limit=1),
            trace_configs=[self._create_trace_config()] if debug else [])
        if debug:
            logger.setLevel(logging.DEBUG)

    def _create_trace_config(self):
        trace_config = aiohttp.TraceConfig()
        trace_config.on_request_start.append(self.on_request_start)
        trace_config.on_request_chunk_sent.append(self.on_request_chunk_sent)
        trace_config.on_response_chunk_received.append(
            self.on_response_chunk_received)
        return trace_config

    async def __aenter__(self):
        pass

    async def __aexit__(self, exc_type, exc, tb):
        await self.client.close()

    async def _handle_stream(self, response):
        is_error = False
        async for line in response.content:
            line = line.decode('utf8').rstrip('\n\r')
            if self.debug:
                logger.info(line)
            sse_msg = ServerSentEvent.decode(line)
            if sse_msg:
                if sse_msg.event == 'error':
                    is_error = True
                if sse_msg.data:
                    if sse_msg.data.startswith('[DONE]'):
                        break
                    yield is_error, response.status, sse_msg.data

    async def _handle_response(self, response: aiohttp.ClientResponse):
        response_status = response.status
        response_content_type = response.content_type
        content_type_json = 'application/json'
        content_type_event_stream = 'text/event-stream'
        is_success = response_status == HTTPStatus.OK

        if is_success:
            # Handle successful response with 'text/event-stream' content type
            if content_type_event_stream in response_content_type:
                async for is_error, status_code, data in self._handle_stream(
                        response):
                    yield (is_error, status_code, data)
            # Handle successful response with 'application/json' content type
            elif content_type_json in response_content_type:
                content = await response.json()
                if content.get('object') == 'error':
                    yield (True, content.get('code'), content.get('message'))
                else:
                    yield (False, HTTPStatus.OK,
                           json.dumps(content, ensure_ascii=False))
            # Handle other successful responses
            else:
                content = await response.read()
                yield (False, HTTPStatus.OK, content)
        else:
            # Handle error response with 'application/json' content type
            if content_type_json in response_content_type:
                error = await response.json()
                yield (True, response_status,
                       json.dumps(error, ensure_ascii=False))
            # Handle error response with 'text/event-stream' content type
            elif content_type_event_stream in response_content_type:
                async for _, _, data in self._handle_stream(response):
                    error = json.loads(data)
                yield (True, response_status,
                       json.dumps(error, ensure_ascii=False))
            # Handle other error responses
            else:
                msg = await response.read()
                yield (True, response_status, msg.decode('utf-8'))

    async def post(self, body):
        headers = {'Content-Type': 'application/json', **self.headers}
        try:
            async with self.client.request(
                    'POST', url=self.url, json=body,
                    headers=headers) as response:
                async for rsp in self._handle_response(response):
                    yield rsp
        except (aiohttp.ClientConnectorError, Exception) as e:
            logger.error(e)
            raise

    @staticmethod
    async def on_request_start(session, context, params):
        logger.info(f'Starting request: <{params}>')

    @staticmethod
    async def on_request_chunk_sent(session, context, params):
        logger.info(f'Request body: {params}')

    @staticmethod
    async def on_response_chunk_received(session, context, params):
        logger.info(f'Response info: <{params}>')
