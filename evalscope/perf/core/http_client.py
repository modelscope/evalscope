import asyncio
import time
from typing import TYPE_CHECKING, Any

import aiohttp

from evalscope.perf.arguments import Arguments
from evalscope.perf.utils.benchmark_util import BenchmarkData, is_stream_body
from evalscope.utils.argument_utils import SECRET_HEADER_KEYS, get_secret_value
from evalscope.utils.logger import get_logger

if TYPE_CHECKING:
    from evalscope.perf.plugin.api.base import ApiPluginBase

logger = get_logger()

# Fixed preflight policy for the pre-benchmark connection test. This is an internal
# health check, not a tunable: a misconfigured URL must fail in well under a minute
# rather than wait for --total-timeout (default 6h). Patch these constants in tests
# to shorten the waits.
_PREFLIGHT_DEADLINE = 60  # total wall-clock seconds for the whole probe
_PREFLIGHT_MAX_ATTEMPTS = 3  # hard cap on attempts regardless of time left
_PREFLIGHT_RETRY_INTERVAL = 2  # seconds between attempts (clamped to remaining time)


class AioHttpClient:
    def __init__(
        self,
        args: Arguments,
        api_plugin: 'ApiPluginBase',
    ):
        self.url = args.url
        self.headers = {'user-agent': 'modelscope_bench', **self._get_runtime_headers(args.headers or {})}
        self.total_timeout = args.total_timeout
        self.read_timeout = args.read_timeout
        self.connect_timeout = args.connect_timeout
        self.api_plugin = api_plugin

        # Configure connector similar to vLLM bench for better TTFT under load.
        _conn_limit = 0 if (args.parallel is None or args.parallel <= 0) else args.parallel
        connector = aiohttp.TCPConnector(
            limit=_conn_limit,  # 0 means no limit in aiohttp; use parallel as limit if set
            limit_per_host=_conn_limit,
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
            trace_configs=[self._create_trace_config()] if args.debug else [],
        )

    @staticmethod
    def _get_runtime_headers(headers: dict) -> dict:
        return {key: get_secret_value(value) for key, value in headers.items()}

    async def __aenter__(self) -> 'AioHttpClient':
        return self

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        if not self.client.closed:
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
        start_time = time.perf_counter()
        try:
            headers, request_id = self.api_plugin.extract_body_meta(body, self.headers)
            # Delegate the request processing to the API plugin
            output = await self.api_plugin.process_request(self.client, self.url, headers, body)
            if not output.success:
                if output.start_time <= 0:
                    output.start_time = start_time
                if output.completed_time < output.start_time:
                    output.completed_time = time.perf_counter()
            if request_id:
                output.request_id = request_id
            return output
        except asyncio.TimeoutError as e:
            logger.error(
                f'TimeoutError: total_timeout: {self.total_timeout}, connect_timeout: {self.connect_timeout}, read_timeout: {self.read_timeout}. Please set longer timeout.'  # noqa: E501
            )
            return self._failure_record(body, str(e), start_time)
        except (aiohttp.ClientConnectorError, Exception) as e:
            logger.error(e)
            return self._failure_record(body, str(e), start_time)

    @staticmethod
    def _failure_record(body, error: str, start_time: float) -> BenchmarkData:
        """Build a BenchmarkData for a request that never produced a response.

        The start time is captured before request preparation so failures cover
        the same lifecycle and use the same ``perf_counter()`` clock as successful
        requests.
        """
        return BenchmarkData(
            success=False,
            error=error,
            is_stream=is_stream_body(body),
            start_time=start_time,
            completed_time=time.perf_counter(),
        )

    @staticmethod
    async def on_request_start(session, context, params: aiohttp.TraceRequestStartParams):
        headers = {
            key: '**********' if str(key).lower() in SECRET_HEADER_KEYS else value
            for key, value in params.headers.items()
        }
        logger.debug(f'Starting request: <method={params.method}, url={params.url}, headers={headers}>')

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
    # Fixed internal preflight policy: a short total deadline, a small attempt cap, and a
    # short retry interval. A misconfigured URL must fail in well under a minute rather than
    # wait for --total-timeout (default 6h). --total-timeout stays scoped to a single request;
    # --no-test-connection remains the escape hatch to skip the probe entirely.
    deadline = time.monotonic() + _PREFLIGHT_DEADLINE

    # Building the request is deterministic and network-independent: a failure here
    # is a configuration error that retrying can never fix, so it happens once,
    # outside the retry loop, and the actionable error is allowed to propagate.
    messages = [{'role': 'user', 'content': 'hello'}] if args.apply_chat_template else 'hello'
    request = api_plugin.build_request(messages)
    if request is None:
        logger.error(
            f'{type(api_plugin).__name__}.build_request() returned None, so the connection test '
            'has nothing to send. Please check the request options this API plugin requires.'
        )
        return False

    async def attempt_connection():
        client = AioHttpClient(args, api_plugin)
        async with client:
            output = await client.post(request)
            return output

    def _log_failure(reason: str) -> None:
        logger.error(
            f'Connection test failed: {reason}. '
            f'Please check --url, --api and --model (or use --no-test-connection to skip).'
        )

    for attempt in range(1, _PREFLIGHT_MAX_ATTEMPTS + 1):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            _log_failure(f'deadline of {_PREFLIGHT_DEADLINE}s exceeded before attempt {attempt}')
            return False
        try:
            # Cap each attempt at the remaining time so a pending request is cancelled at the
            # deadline instead of waiting for --total-timeout.
            output = await asyncio.wait_for(attempt_connection(), timeout=remaining)
        except asyncio.TimeoutError:
            _log_failure(f'request hung past the {_PREFLIGHT_DEADLINE}s deadline')
            return False
        except Exception as e:
            # Network error: retry if attempts and deadline allow.
            logger.warning(f'Retrying... <{e}>')
        else:
            if output.success:
                logger.info('Test connection successful.')
                return True
            # 4xx errors are client-side configuration errors (wrong URL, wrong request format,
            # auth failure, etc.) and will never succeed on retry — bail out immediately.
            if output.status_code is not None and 400 <= output.status_code < 500:
                _log_failure(f'non-retryable HTTP {output.status_code}: {output.error}')
                return False
            # Retryable 5xx: retry if attempts and deadline allow.
            logger.warning(f'Retrying... <{output.error}>')

        if attempt < _PREFLIGHT_MAX_ATTEMPTS:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                _log_failure(f'deadline of {_PREFLIGHT_DEADLINE}s exceeded after attempt {attempt}')
                return False
            await asyncio.sleep(min(_PREFLIGHT_RETRY_INTERVAL, remaining))

    _log_failure(f'exhausted {_PREFLIGHT_MAX_ATTEMPTS} attempts within {_PREFLIGHT_DEADLINE}s')
    return False
