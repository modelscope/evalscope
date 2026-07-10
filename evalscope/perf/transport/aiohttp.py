from __future__ import annotations

import aiohttp
import json
import time
from typing import AsyncIterator, Callable, Optional

from evalscope.perf.config.models import TargetConfig
from evalscope.perf.domain.errors import TransportError
from evalscope.perf.domain.observation import TransportEvent
from evalscope.perf.transport.base import HttpRequest, HttpTransport
from evalscope.perf.transport.sse import SSEDecoder


class AioHttpTransport(HttpTransport):
    """Shared aiohttp transport for every supported API protocol."""

    def __init__(
        self,
        target: TargetConfig,
        connector_limit: int = 0,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        self._target = target
        self._connector_limit = connector_limit
        self._clock = clock
        self._session: Optional[aiohttp.ClientSession] = None

    async def __aenter__(self) -> 'AioHttpTransport':
        connector = aiohttp.TCPConnector(
            limit=self._connector_limit,
            limit_per_host=self._connector_limit,
            ttl_dns_cache=300,
            use_dns_cache=True,
            keepalive_timeout=60,
            enable_cleanup_closed=True,
            force_close=False,
        )
        timeout = aiohttp.ClientTimeout(
            total=self._target.total_timeout,
            connect=self._target.connect_timeout,
            sock_read=self._target.read_timeout,
        )
        self._session = aiohttp.ClientSession(connector=connector, trust_env=True, timeout=timeout)
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        if self._session is not None:
            await self._session.close()
            self._session = None

    async def events(self, request: HttpRequest) -> AsyncIterator[TransportEvent]:
        if self._session is None:
            raise TransportError('HTTP transport is not open')

        headers = {'Content-Type': 'application/json', 'User-Agent': 'evalscope-perf', **request.headers}
        try:
            start = self._clock()
            yield TransportEvent(kind='request_start', timestamp=start)
            async with self._session.post(request.url, json=request.body, headers=headers) as response:
                content_type = response.headers.get('Content-Type', '')
                yield TransportEvent(
                    kind='response_start',
                    timestamp=self._clock(),
                    status_code=response.status,
                    content_type=content_type,
                )
                if response.status >= 400:
                    raw = await response.text()
                    try:
                        payload = json.loads(raw)
                    except json.JSONDecodeError:
                        payload = raw
                    yield TransportEvent(
                        kind='http_error',
                        timestamp=self._clock(),
                        status_code=response.status,
                        data=payload,
                    )
                    return

                if 'text/event-stream' in content_type:
                    decoder = SSEDecoder()
                    async for chunk in response.content.iter_any():
                        if not chunk:
                            continue
                        for message in decoder.feed(chunk):
                            yield TransportEvent(kind='sse', timestamp=self._clock(), data=message)
                    for message in decoder.finish():
                        yield TransportEvent(kind='sse', timestamp=self._clock(), data=message)
                else:
                    raw = await response.text()
                    try:
                        payload = json.loads(raw)
                    except json.JSONDecodeError:
                        payload = raw
                    yield TransportEvent(kind='json', timestamp=self._clock(), data=payload)
                yield TransportEvent(kind='response_end', timestamp=self._clock())
        except aiohttp.ClientError as e:
            raise TransportError(str(e)) from e
