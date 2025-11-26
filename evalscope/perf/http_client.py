import aiohttp
import asyncio
import time
from typing import TYPE_CHECKING, AsyncGenerator, Dict, List, Tuple

from evalscope.perf.utils.benchmark_util import BenchmarkData
from evalscope.utils.logger import get_logger
from .arguments import Arguments

if TYPE_CHECKING:
    from .plugin.api.base import ApiPluginBase

logger = get_logger()


class AioHttpClient:

    def __init__(
        self,
        args: Arguments,
        api_plugin: 'ApiPluginBase',
    ):
        self.url = args.url
        self.headers = {'user-agent': 'modelscope_bench', **(args.headers or {})}
        self.total_timeout = args.total_timeout
        self.read_timeout = args.read_timeout
        self.connect_timeout = args.connect_timeout
        self.api_plugin = api_plugin

        # Configure connector similar to vLLM bench for better TTFT under load.
        connector = aiohttp.TCPConnector(
            limit=args.parallel or 0,  # 0 means no limit in aiohttp; use parallel as limit if set
            limit_per_host=args.parallel or 0,
            ttl_dns_cache=300,
            use_dns_cache=True,
            keepalive_timeout=60,
            enable_cleanup_closed=True,
            force_close=False,
            ssl=('https://' in self.url),
        )

        client_timeout = aiohttp.ClientTimeout(
            total=self.total_timeout, connect=self.connect_timeout, sock_read=self.read_timeout
        )

        self.client = aiohttp.ClientSession(
            connector=connector,
            trust_env=True,
            timeout=client_timeout,
            trace_configs=[self._create_trace_config()] if args.debug else []
        )

    async def __aenter__(self):
        pass

    async def __aexit__(self, exc_type, exc, tb):
        await self.client.close()

    def _create_trace_config(self):
        """Create trace configuration for debugging."""
        trace_config = aiohttp.TraceConfig()
        trace_config.on_request_start.append(self.on_request_start)
        trace_config.on_request_chunk_sent.append(self.on_request_chunk_sent)
        trace_config.on_response_chunk_received.append(self.on_response_chunk_received)
        return trace_config

    async def post(self, body) -> BenchmarkData:
        """
        Send POST request and delegate response handling to API plugin.

        Returns:
            BenchmarkData: The benchmark data object containing request and response information.
        """
        try:
            # Delegate the request processing to the API plugin
            output = await self.api_plugin.process_request(self.client, self.url, self.headers, body)
            return output
        except asyncio.TimeoutError as e:
            logger.error(
                f'TimeoutError: total_timeout: {self.total_timeout}, connect_timeout: {self.connect_timeout}, read_timeout: {self.read_timeout}. Please set longer timeout.'  # noqa: E501
            )
            return BenchmarkData(success=False, error=str(e))
        except (aiohttp.ClientConnectorError, Exception) as e:
            logger.error(e)
            return BenchmarkData(success=False, error=str(e))

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


async def test_connection(args: Arguments, api_plugin: 'ApiPluginBase') -> bool:
    start_time = time.perf_counter()

    async def attempt_connection():
        client = AioHttpClient(args, api_plugin)
        async with client:
            messages = [{'role': 'user', 'content': 'hello'}] if args.apply_chat_template else 'hello'
            request = api_plugin.build_request(messages)

            output = await client.post(request)
            return output

    while True:
        try:
            output = await attempt_connection()
            if output.success:
                logger.info('Test connection successful.')
                return True
            logger.warning(f'Retrying... <{output.error}>')
        except Exception as e:
            logger.warning(f'Retrying... <{e}>')

        if time.perf_counter() - start_time >= args.total_timeout:
            logger.error('Overall connection attempt timed out.')
            return False

        await asyncio.sleep(10)
