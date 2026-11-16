import asyncio
import urllib.request
from pathlib import Path

import aiohttp.helpers
import pytest
from aiohttp import BasicAuth, web

from evalscope.perf.arguments import Arguments
from evalscope.perf.core.http_client import AioHttpClient, _has_request_environment

PROXY_VARS = ['http_proxy', 'https_proxy', 'all_proxy', 'no_proxy']
URL = 'http://127.0.0.1:8000/v1/chat/completions'


@pytest.fixture
def netrc_path(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Clear the proxy variables and point NETRC at a file that does not exist yet."""
    for name in PROXY_VARS:
        monkeypatch.delenv(name, raising=False)
        monkeypatch.delenv(name.upper(), raising=False)
    # On macOS and Windows, getproxies() also reads the system proxy. Read only the variables.
    monkeypatch.setattr(urllib.request, 'getproxies', urllib.request.getproxies_environment)
    monkeypatch.setattr(urllib.request, 'proxy_bypass', urllib.request.proxy_bypass_environment)
    monkeypatch.setattr(aiohttp.helpers, 'getproxies', urllib.request.getproxies_environment)
    monkeypatch.setattr(aiohttp.helpers, 'proxy_bypass', urllib.request.proxy_bypass_environment)
    path = tmp_path / 'netrc'
    monkeypatch.setenv('NETRC', str(path))
    return path


def test_no_proxy_and_no_netrc(netrc_path: Path) -> None:
    assert not _has_request_environment()


def test_proxy_preserved_for_other_schemes(netrc_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('HTTP_PROXY', 'http://proxy.internal:3128')
    assert _has_request_environment()
    assert _client_trusts_env('https://10.0.0.5:8000/v1/chat/completions')


def test_proxy_preserved_when_initial_host_is_bypassed(netrc_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('HTTP_PROXY', 'http://proxy.internal:3128')
    monkeypatch.setenv('NO_PROXY', 'localhost,127.0.0.1')
    assert _has_request_environment()
    assert _client_trusts_env(URL)


def test_netrc_entry_preserved_for_other_hosts(netrc_path: Path) -> None:
    netrc_path.write_text('machine api.example.com login user password secret\n')
    assert _has_request_environment()
    assert _client_trusts_env(URL)


@pytest.mark.parametrize('content', ['', 'machine', 'default login user password secret\n'])
def test_empty_malformed_and_default_netrc(netrc_path: Path, content: str) -> None:
    netrc_path.write_text(content)
    assert _has_request_environment() == content.startswith('default')


def _client_trusts_env(url: str) -> bool:
    async def run() -> bool:
        args = Arguments(model='test-model', api='openai', url=url, number=1, parallel=1)
        # Arguments keeps sweep lists. run_one_benchmark sets scalars before it creates the client.
        args.number = 1
        args.parallel = 1
        async with AioHttpClient(args, None) as client:
            return client.client.trust_env

    return asyncio.run(run())


def test_client_does_not_trust_env_without_proxy_or_netrc(netrc_path: Path) -> None:
    assert not _client_trusts_env(URL)


def test_client_trusts_env_when_a_proxy_applies(netrc_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('HTTP_PROXY', 'http://proxy.internal:3128')
    assert _client_trusts_env('http://10.0.0.5:8000/v1/chat/completions')


def test_redirect_uses_destination_netrc_credentials(netrc_path: Path) -> None:
    netrc_path.write_text('machine localhost login user password secret\n')

    async def run() -> None:
        app = web.Application()

        async def redirect(request: web.Request) -> web.Response:
            return web.Response(status=307, headers={'Location': f'http://localhost:{port}/target'})

        async def target(request: web.Request) -> web.Response:
            authorized = request.headers.get('Authorization') == BasicAuth('user', 'secret').encode()
            return web.Response(status=200 if authorized else 401)

        app.router.add_post('/v1/chat/completions', redirect)
        app.router.add_post('/target', target)
        runner = web.AppRunner(app)
        await runner.setup()
        try:
            site = web.TCPSite(runner, '127.0.0.1', 0)
            await site.start()
            port = site._server.sockets[0].getsockname()[1]
            args = Arguments(model='test', url=f'http://127.0.0.1:{port}/v1/chat/completions', number=1, parallel=1)
            args.number = args.parallel = 1
            async with AioHttpClient(args, None) as client:
                async with client.client.post(args.url, json={}) as response:
                    assert response.status == 200
                    assert len(response.history) == 1
        finally:
            await runner.cleanup()

    asyncio.run(run())
