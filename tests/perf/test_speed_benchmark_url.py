"""speed_benchmark sends raw-text prompts, so every request must reach the completions endpoint."""

import asyncio
import json
from pathlib import Path
from typing import List

import pytest

from evalscope.perf.arguments import Arguments
from evalscope.perf.benchmark import run_benchmark
from evalscope.perf.plugin.api.openai_api import OpenaiPlugin
from evalscope.perf.utils.benchmark_util import BenchmarkData


def _run(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, url: str) -> List[str]:
    urls: List[str] = []

    async def process_request(self, client_session, request_url, headers, body) -> BenchmarkData:
        urls.append(request_url)
        assert 'prompt' in body
        return BenchmarkData(
            request=json.dumps(body),
            success=True,
            start_time=1.0,
            completed_time=2.0,
            query_latency=1.0,
            first_chunk_latency=0.5,
            prompt_tokens=1,
            completion_tokens=2,
        )

    monkeypatch.setattr(OpenaiPlugin, 'process_request', process_request)
    args = Arguments(
        model='test-model',
        url=url,
        dataset='speed_benchmark',
        number=2,
        parallel=1,
        no_test_connection=True,
        outputs_dir=str(tmp_path),
    )
    args.number, args.parallel, args.rate = 2, 1, -1
    asyncio.run(run_benchmark(args))
    return urls


@pytest.mark.parametrize('url', ['http://127.0.0.1:1/v1/chat/completions', 'http://127.0.0.1:1/v1'])
def test_requests_reach_completions_endpoint(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, url: str) -> None:
    assert _run(monkeypatch, tmp_path, url) == ['http://127.0.0.1:1/v1/completions'] * 2


def test_connection_probe_matches_the_completions_endpoint() -> None:
    args = Arguments(model='test-model', url='http://127.0.0.1:1/v1/chat/completions', dataset='speed_benchmark')

    assert args.url == 'http://127.0.0.1:1/v1/completions'
    # The probe sends a chat message only when this is on; /v1/completions needs a raw prompt.
    assert args.apply_chat_template is False


def test_other_datasets_keep_the_chat_endpoint() -> None:
    args = Arguments(model='test-model', url='http://127.0.0.1:1/v1/chat/completions', dataset='openqa')

    assert args.url == 'http://127.0.0.1:1/v1/chat/completions'
    assert args.apply_chat_template is True
