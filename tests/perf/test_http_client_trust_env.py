import asyncio
import urllib.request
from pathlib import Path

import pytest

from evalscope.perf.arguments import Arguments
from evalscope.perf.core.http_client import AioHttpClient, _env_changes_request

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
    path = tmp_path / 'netrc'
    monkeypatch.setenv('NETRC', str(path))
    return path


def test_no_proxy_and_no_netrc(netrc_path: Path) -> None:
    assert not _env_changes_request(URL)


def test_proxy_for_the_scheme(netrc_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('HTTP_PROXY', 'http://proxy.internal:3128')
    assert _env_changes_request('http://10.0.0.5:8000/v1/chat/completions')
    assert not _env_changes_request('https://10.0.0.5:8000/v1/chat/completions')


def test_no_proxy_excludes_the_host(netrc_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('HTTP_PROXY', 'http://proxy.internal:3128')
    monkeypatch.setenv('NO_PROXY', 'localhost,127.0.0.1')
    assert not _env_changes_request(URL)


def test_netrc_entry_for_the_host(netrc_path: Path) -> None:
    netrc_path.write_text('machine api.example.com login user password secret\n')
    assert _env_changes_request('https://api.example.com/v1/chat/completions')
    assert not _env_changes_request(URL)


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
