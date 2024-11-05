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
        client_timeout = aiohttp.ClientTimeout(
            total=read_timeout + conn_timeout,
            connect=conn_timeout,
            sock_read=read_timeout,
        )
        self.debug = debug
        if debug:
            logger.setLevel(level=logging.DEBUG)
            trace_config = aiohttp.TraceConfig()
            trace_config.on_request_start.append(self.on_request_start)
            trace_config.on_request_chunk_sent.append(
                self.on_request_chunk_sent)
            trace_config.on_response_chunk_received.append(
                self.on_response_chunk_received)
        self.client = aiohttp.ClientSession(
            trace_configs=[trace_config] if debug else [],
            connector=aiohttp.TCPConnector(limit=1),
            timeout=client_timeout,
        )
        ua = 'modelscope_bench'
        self.headers = {'user-agent': ua}
        if headers:
            self.headers.update(headers)
        self.url = url

    async def __aenter__(self):
        pass

    async def __aexit__(self, exc_type, exc, tb):
        await self.client.close()

    async def _handle_stream(self, response):
        is_error = False
        status_code = response.status
        async for line in response.content:
            if line:
                line = line.decode('utf8')
                line = line.rstrip('\n').rstrip('\r')
                if self.debug:
                    logger.info(line)
                sse_msg = ServerSentEvent.decode(line)
                if not sse_msg:
                    continue
                if sse_msg.event and sse_msg.event == 'error':  # dashscope error
                    is_error = True

                if sse_msg.data:
                    if sse_msg.data.startswith(
                            '[DONE]'):  # openai api completed
                        break
                    yield (is_error, status_code, sse_msg.data)
                    # yield data

    async def _handle_response(self, response: aiohttp.ClientResponse):
        if (response.status == HTTPStatus.OK
                and 'text/event-stream' in response.content_type):
            async for is_error, status_code, data in self._handle_stream(
                    response):
                yield (is_error, status_code, data)
        elif (response.status == HTTPStatus.OK
              and 'application/json' in response.content_type):
            content = await response.json()
            if 'object' in content and content['object'] == 'error':
                yield (True, content['code'], content['message'])
            else:
                yield (False, HTTPStatus.OK,
                       json.dumps(content, ensure_ascii=False))
        elif response.status == HTTPStatus.OK:
            content = await response.read()
            yield (False, HTTPStatus.OK, content)
        else:
            if 'application/json' in response.content_type:
                error = await response.json()
                yield (True, response.status,
                       json.dumps(error, ensure_ascii=False))
            elif 'text/event-stream' in response.content_type:
                async for _, _, data in self._handle_stream(response):
                    error = json.loads(data)
                yield (True, response.status, error)
            else:
                msg = await response.read()
                yield (True, response.status, msg.decode('utf-8'))

    async def post(self, body):
        try:
            headers = {'Content-Type': 'application/json', **self.headers}
            response = await self.client.request(
                'POST', url=self.url, json=body, headers=headers)
            async with response:
                async for rsp in self._handle_response(response):
                    yield rsp
        except aiohttp.ClientConnectorError as e:
            logger.error(e)
            raise e
        except Exception as e:
            logger.error(e)
            raise e

    @staticmethod
    async def on_request_start(session, context, params):
        logger.info(f'Starting request: <{params}>')

    @staticmethod
    async def on_request_chunk_sent(session, context, params):
        logger.info(f'Request body: {params}')

    @staticmethod
    async def on_response_chunk_received(session, context, params):
        logger.info(f'Response info: <{params}>')
