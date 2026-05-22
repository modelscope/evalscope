"""In-process aiohttp server that proxies LLM traffic from external
agents to EvalScope's model layer.

The server is started lazily (first ``trial_session()``) and reused
across samples / runs in the same Python process.  Each concurrent run
gets its own :class:`TrialSession` keyed by a bearer token that the
runner injects into the agent's environment.
"""

import asyncio
import json
import time
import uuid
from aiohttp import web
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any, AsyncIterator, Dict, Optional
from urllib.parse import urlsplit, urlunsplit

from evalscope.api.model import GenerateConfig, Model
from evalscope.utils.function_utils import AsyncioLoopRunner
from evalscope.utils.logger import get_logger
from ..runners.base import BridgeEndpoint
from .sse_anthropic import stream_anthropic_response
from .trace_recorder import BridgeTraceRecorder
from .translate_anthropic import (
    anthropic_request_to_messages,
    anthropic_tools_to_tool_infos,
    model_output_to_anthropic_response,
)

if TYPE_CHECKING:
    from evalscope.api.agent import AgentEnvironment

logger = get_logger()

_TRIAL_TOKEN_PREFIX = 'trial-'

#: Env names whose runtime is a Docker container — for these we rewrite
#: the bridge's loopback host to ``host.docker.internal`` so the agent
#: inside the container can reach the bridge on the host. Shared with
#: :mod:`evalscope.agent.external.adapter`.
DOCKER_ENV_NAMES = frozenset({'enclave', 'docker', 'volcengine'})

#: Loopback hosts the bridge may bind to — these are the strings that
#: have no meaning inside a container and must be translated.
_LOOPBACK_HOSTS = frozenset({'127.0.0.1', '0.0.0.0', 'localhost', '::1'})


def _rewrite_for_env(base_url: str, env: 'Optional[AgentEnvironment]') -> str:
    """Return ``base_url`` with the host swapped for a value the agent
    can dial from inside ``env``.

    No-op when ``env`` is ``None`` (caller did not opt in), when the env
    is not a Docker variant, or when the bound host is already routable
    from the container (anything outside the loopback set). Otherwise
    swap to ``host.docker.internal`` — Docker Desktop provides it natively
    on macOS / Windows; on Linux the adapter must inject an
    ``extra_hosts: host.docker.internal:host-gateway`` mapping into the
    sandbox config (see :mod:`evalscope.agent.external.adapter`).
    """
    if env is None:
        return base_url
    env_name = getattr(env, 'name', '') or ''
    if env_name not in DOCKER_ENV_NAMES:
        return base_url
    parts = urlsplit(base_url)
    if parts.hostname not in _LOOPBACK_HOSTS:
        return base_url
    new_netloc = f'host.docker.internal:{parts.port}' if parts.port else 'host.docker.internal'
    return urlunsplit((parts.scheme, new_netloc, parts.path, parts.query, parts.fragment))


class TrialSession:
    """Per-run handle returned by :meth:`ModelProxyServer.trial_session`."""

    def __init__(
        self,
        trial_id: str,
        token: str,
        base_url: str,
        model: Model,
        recorder: BridgeTraceRecorder,
        framework: str,
    ) -> None:
        self.trial_id = trial_id
        self.token = token
        self.base_url = base_url
        self.model = model
        self.recorder = recorder
        self.framework = framework

    def endpoint_view(self, for_env: 'Optional[AgentEnvironment]' = None) -> BridgeEndpoint:
        """Return the runner-facing view (base_url + trial_token).

        When ``for_env`` is supplied and resolves to a Docker-backed
        environment, the loopback host in ``base_url`` is swapped for
        ``host.docker.internal`` so the agent inside the container can
        reach the bridge running on the host. Pass ``None`` (the default)
        to inherit the host the bridge actually bound to — used by the
        Local environment, where the runner shares the host network.
        """
        base_url = _rewrite_for_env(self.base_url, for_env)
        return BridgeEndpoint(base_url=base_url, trial_token=self.token)


class ModelProxyServer:
    """Per-loop aiohttp server that routes ``trial_id`` → :class:`Model`.

    Keyed on ``id(loop)`` rather than process-global so each
    :class:`AsyncioLoopRunner`-owned worker thread gets its own bridge
    (and its own port) — the alternative would be cross-loop calls into
    aiohttp internals, which silently break the moment a second worker
    thread tries to use the bridge.

    The instance auto-shuts down when its loop closes via
    :meth:`AsyncioLoopRunner.register_close_callback`.
    """

    _instances: Dict[int, 'ModelProxyServer'] = {}
    _instances_lock = asyncio.Lock()

    def __init__(self, host: str, port: Optional[int]) -> None:
        self._host = host
        self._configured_port = port
        self._actual_port: Optional[int] = None
        self._app: Optional[web.Application] = None
        self._runner: Optional[web.AppRunner] = None
        self._site: Optional[web.TCPSite] = None
        self._sessions: Dict[str, TrialSession] = {}
        self._sessions_lock = asyncio.Lock()
        self._started = False
        self._start_lock = asyncio.Lock()

    @classmethod
    async def get_or_start(
        cls,
        host: str = '127.0.0.1',
        port: Optional[int] = None,
    ) -> 'ModelProxyServer':
        """Return the per-loop singleton, starting it on first call.

        Each event loop (typically one per worker thread via
        :class:`AsyncioLoopRunner`) gets its own instance — bridges bound
        to a different loop are unreachable from here anyway.
        """
        loop_key = id(asyncio.get_running_loop())
        async with cls._instances_lock:
            inst = cls._instances.get(loop_key)
            if inst is None:
                inst = cls(host=host, port=port)
                cls._instances[loop_key] = inst
        await inst._ensure_started()
        return inst

    async def _ensure_started(self) -> None:
        if self._started:
            return
        async with self._start_lock:
            if self._started:
                return
            app = web.Application()
            app.router.add_post('/anthropic/v1/messages', self._handle_anthropic_messages)

            async def _healthz(_request: web.Request) -> web.Response:
                return web.json_response({'ok': True})

            app.router.add_get('/healthz', _healthz)

            async def _head_probe(_request: web.Request) -> web.Response:
                # Anthropic SDKs (claude-code among them) issue an
                # endpoint reachability HEAD before the first POST. A
                # HEAD has no body per HTTP semantics, so 200 with no
                # payload is the right answer and avoids polluting the
                # bridge log with one-off "unhandled HEAD" warnings on
                # every fresh trial session.
                return web.Response(status=200)

            app.router.add_route('HEAD', '/{tail:.*}', _head_probe)

            async def _catchall(request: web.Request) -> web.Response:
                logger.warning(f'bridge: unhandled {request.method} {request.path} '
                               f'(query={dict(request.query)})')
                return web.json_response(
                    {
                        'type': 'error',
                        'error': {
                            'type': 'not_found',
                            'message': f'no handler for {request.path}'
                        }
                    },
                    status=404,
                )

            app.router.add_route('*', '/{tail:.*}', _catchall)
            runner = web.AppRunner(app, access_log=None)
            await runner.setup()
            site = web.TCPSite(runner, host=self._host, port=self._configured_port or 0)
            await site.start()
            # Resolve the actual port (handles port=0 / auto-pick).
            self._actual_port = self._resolve_actual_port(site)
            self._app = app
            self._runner = runner
            self._site = site
            self._started = True
            # Auto-release the port when the owning loop shuts down so we don't
            # leak the singleton across pytest sessions / worker tear-down.
            AsyncioLoopRunner.register_close_callback(self.shutdown)
            logger.info(
                f'ModelProxyServer started on http://{self._host}:{self._actual_port} '
                f'(Anthropic route: /anthropic/v1/messages)'
            )

    @staticmethod
    def _resolve_actual_port(site: web.TCPSite) -> int:
        # aiohttp TCPSite exposes the underlying server's sockets via _server.
        server = site._server  # type: ignore[attr-defined]
        if server is not None and server.sockets:
            return server.sockets[0].getsockname()[1]
        return 0

    @property
    def base_url(self) -> str:
        if not self._started or self._actual_port is None:
            raise RuntimeError('ModelProxyServer has not been started yet.')
        return f'http://{self._host}:{self._actual_port}'

    async def shutdown(self) -> None:
        """Stop the server and release the port.  Safe to call multiple times.

        Invoked automatically by :class:`AsyncioLoopRunner`'s close-callback
        machinery when the owning loop shuts down, but also exposed for
        explicit teardown in tests.
        """
        if not self._started:
            return
        if self._site is not None:
            await self._site.stop()
        if self._runner is not None:
            await self._runner.cleanup()
        self._started = False
        self._site = None
        self._runner = None
        self._app = None
        loop_key = id(asyncio.get_running_loop())
        async with self._instances_lock:
            if ModelProxyServer._instances.get(loop_key) is self:
                ModelProxyServer._instances.pop(loop_key, None)

    @asynccontextmanager
    async def trial_session(
        self,
        model: Model,
        framework: str,
    ) -> AsyncIterator[TrialSession]:
        """Register a trial → model mapping for the duration of the run.

        Yields a :class:`TrialSession` whose ``base_url`` / ``token`` should
        be injected into the agent's environment variables.
        """
        trial_id = uuid.uuid4().hex
        token = f'{_TRIAL_TOKEN_PREFIX}{trial_id}'
        recorder = BridgeTraceRecorder(
            trial_id=trial_id,
            framework=framework,
            model_name=getattr(model, 'name', None),
        )
        session = TrialSession(
            trial_id=trial_id,
            token=token,
            base_url=self.base_url,
            model=model,
            recorder=recorder,
            framework=framework,
        )
        async with self._sessions_lock:
            self._sessions[trial_id] = session
        try:
            yield session
        finally:
            async with self._sessions_lock:
                self._sessions.pop(trial_id, None)

    # ---- routes -------------------------------------------------------

    async def _handle_anthropic_messages(self, request: web.Request) -> web.StreamResponse:
        logger.debug(
            f'bridge: POST /anthropic/v1/messages from {request.remote} '
            f'auth={request.headers.get("Authorization") or request.headers.get("x-api-key")!r}'
        )
        try:
            session = await self._lookup_session(request)
        except _BridgeAuthError as exc:
            logger.debug(f'bridge: auth failed — {exc}')
            return web.json_response(
                {
                    'type': 'error',
                    'error': {
                        'type': 'authentication_error',
                        'message': str(exc)
                    }
                },
                status=401,
            )

        body = await request.json()
        chat_messages = anthropic_request_to_messages(body)
        tool_infos = anthropic_tools_to_tool_infos(body.get('tools') or [])
        gen_config = _build_generate_config(body)

        if body.get('stream'):
            return await self._respond_streaming(request, session, body, chat_messages, tool_infos, gen_config)
        return await self._respond_json(session, body, chat_messages, tool_infos, gen_config)

    async def _respond_json(
        self,
        session: TrialSession,
        body: Dict[str, Any],
        chat_messages: list,
        tool_infos: list,
        gen_config: 'GenerateConfig',
    ) -> web.StreamResponse:
        started = time.monotonic()
        try:
            output = await session.model.generate_async(
                input=chat_messages,
                tools=tool_infos or None,
                config=gen_config,
            )
        except Exception as exc:  # pragma: no cover - upstream-dependent
            logger.exception(f'bridge: model.generate_async failed (trial={session.trial_id})')
            return web.json_response(
                {
                    'type': 'error',
                    'error': {
                        'type': 'api_error',
                        'message': repr(exc)
                    }
                },
                status=502,
            )

        latency_ms = (time.monotonic() - started) * 1000
        session.recorder.record_anthropic_turn(body, output, latency_ms=latency_ms)
        _log_turn(session, output, latency_ms, mode='json')
        return web.json_response(model_output_to_anthropic_response(output, request_model=body.get('model')))

    async def _respond_streaming(
        self,
        request: web.Request,
        session: TrialSession,
        body: Dict[str, Any],
        chat_messages: list,
        tool_infos: list,
        gen_config: 'GenerateConfig',
    ) -> web.StreamResponse:
        """Synthesize Anthropic SSE events from a non-streaming completion.

        The underlying ``Model.generate_async`` call is started as a task so
        the synthesizer can emit ``ping`` events while it runs, then slice
        the final ``ModelOutput`` into ``content_block_*`` events.
        """
        response = web.StreamResponse(
            status=200,
            headers={
                'Content-Type': 'text/event-stream; charset=utf-8',
                'Cache-Control': 'no-cache',
                'Connection': 'keep-alive',
            },
        )
        await response.prepare(request)

        started = time.monotonic()
        generate_task = asyncio.create_task(
            session.model.generate_async(
                input=chat_messages,
                tools=tool_infos or None,
                config=gen_config,
            )
        )
        try:
            async for chunk in stream_anthropic_response(generate_task, request_model=body.get('model')):
                await response.write(chunk)
            # Recorder needs the resolved output; awaiting the task is a no-op
            # because the streamer already drained it.
            output = await generate_task
            latency_ms = (time.monotonic() - started) * 1000
            session.recorder.record_anthropic_turn(body, output, latency_ms=latency_ms)
            _log_turn(session, output, latency_ms, mode='stream')
        except Exception as exc:  # pragma: no cover - upstream-dependent
            logger.exception(f'bridge: streaming generate failed (trial={session.trial_id})')
            error_event = (
                f'event: error\ndata: '
                f'{json.dumps({"type": "error", "error": {"type": "api_error", "message": repr(exc)}})}'
                f'\n\n'
            ).encode('utf-8')
            try:
                await response.write(error_event)
            except ConnectionResetError:
                pass
        finally:
            if not generate_task.done():
                generate_task.cancel()
        await response.write_eof()
        return response

    async def _lookup_session(self, request: web.Request) -> TrialSession:
        token = _extract_bearer_token(request)
        if not token or not token.startswith(_TRIAL_TOKEN_PREFIX):
            raise _BridgeAuthError(f'missing or malformed bridge token (expected {_TRIAL_TOKEN_PREFIX}<id>)')
        trial_id = token[len(_TRIAL_TOKEN_PREFIX):]
        async with self._sessions_lock:
            session = self._sessions.get(trial_id)
        if session is None:
            raise _BridgeAuthError(f'unknown trial_id {trial_id!r}')
        return session


class _BridgeAuthError(Exception):
    pass


def _log_turn(session: 'TrialSession', output, latency_ms: float, *, mode: str) -> None:
    """Emit a one-line INFO heartbeat per upstream LLM turn.

    Without this, a 5–10 min agent run looks indistinguishable from a
    hang in the main eval log. One line per ``model_generate`` round
    trip gives operators ``step / latency / tokens / stop_reason`` at a
    glance and lets them tell ``stuck on network`` from ``running fine``.
    """
    usage = getattr(output, 'usage', None)
    in_tok = getattr(usage, 'input_tokens', 0) if usage else 0
    out_tok = getattr(usage, 'output_tokens', 0) if usage else 0
    logger.info(
        f'bridge[{session.framework}/{session.trial_id[:8]}] '
        f'step={session.recorder.current_step} {mode} '
        f'latency={latency_ms / 1000:.2f}s '
        f'tokens={in_tok}+{out_tok} '
        f'stop={getattr(output, "stop_reason", None)!r}'
    )


def _extract_bearer_token(request: web.Request) -> Optional[str]:
    # claude-code sends ANTHROPIC_AUTH_TOKEN as ``Authorization: Bearer <token>``
    # plus an ``x-api-key`` mirror for compatibility.  Accept either.
    auth = request.headers.get('Authorization') or request.headers.get('authorization')
    if auth and auth.lower().startswith('bearer '):
        return auth.split(None, 1)[1].strip()
    xkey = request.headers.get('x-api-key') or request.headers.get('X-Api-Key')
    return xkey.strip() if xkey else None


def _build_generate_config(body: Dict[str, Any]) -> GenerateConfig:
    """Translate Anthropic generation params into ``GenerateConfig``.

    Forces ``stream=True`` upstream regardless of the client's choice:
    claude-code sends ``max_tokens=32000`` which trips the SDK's
    ``Streaming is required ...`` guard otherwise. The client-facing
    response shape is decided independently from ``body.get('stream')``.
    """
    kwargs: Dict[str, Any] = {'stream': True}
    if 'max_tokens' in body:
        kwargs['max_tokens'] = body['max_tokens']
    if 'temperature' in body:
        kwargs['temperature'] = body['temperature']
    if 'top_p' in body:
        kwargs['top_p'] = body['top_p']
    if 'stop_sequences' in body and body['stop_sequences']:
        kwargs['stop_seqs'] = list(body['stop_sequences'])
    return GenerateConfig(**kwargs)


__all__ = ['ModelProxyServer', 'TrialSession']
