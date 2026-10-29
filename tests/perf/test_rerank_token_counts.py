"""Rerank billing units must not become token counts in benchmark reports."""

import asyncio
import json
from typing import Any

import aiohttp
import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer

from evalscope.perf.arguments import Arguments
from evalscope.perf.plugin.api.openai_rerank_api import OpenaiRerankPlugin
from evalscope.perf.utils.benchmark_util import MetricsAccumulator


class WordTokenizer:
    def encode(self, text: str, add_special_tokens: bool = False) -> list[str]:
        return text.split()


@pytest.mark.parametrize('search_units', [1, 2, 0])
def test_search_units_do_not_override_tokenizer(search_units: int) -> None:
    async def run() -> None:
        payload = {
            'results': [{'index': 0, 'relevance_score': 0.9}],
            'meta': {'billed_units': {'search_units': search_units}},
        }

        async def handle(request: web.Request) -> web.Response:
            assert await request.json() == {
                'query': 'two words',
                'documents': ['three more words', 'four words in total'],
                'model': 'test-rerank',
            }
            return web.json_response(payload)

        app = web.Application()
        app.router.add_post('/rerank', handle)
        plugin = OpenaiRerankPlugin(Arguments(model='test-rerank'))
        plugin.tokenizer = WordTokenizer()
        body = plugin.build_request({'query': 'two words', 'documents': ['three more words', 'four words in total']})
        async with TestServer(app) as server, aiohttp.ClientSession() as session:
            output = await plugin.process_request(session, str(server.make_url('/rerank')), {}, body)

        accumulator = MetricsAccumulator()
        accumulator.update(output, plugin)
        assert output.success
        assert output.prompt_tokens == 9
        assert output.completion_tokens == 0
        assert accumulator.to_result().avg_prompt_tokens == 9

    asyncio.run(run())


@pytest.mark.parametrize('with_tokenizer', [False, True])
def test_search_units_without_request_tokens(with_tokenizer: bool) -> None:
    plugin = OpenaiRerankPlugin(Arguments(model='test-rerank'))
    if with_tokenizer:
        plugin.tokenizer = WordTokenizer()
    response = {'meta': {'billed_units': {'search_units': 1}}}
    assert plugin.parse_responses([response]) == (0, 0)


@pytest.mark.parametrize('usage', [{'prompt_tokens': 42}, {'total_tokens': 42}])
def test_reported_token_usage_takes_precedence(usage: dict[str, Any]) -> None:
    plugin = OpenaiRerankPlugin(Arguments(model='test-rerank'))
    plugin.tokenizer = WordTokenizer()
    response = {'usage': usage, 'meta': {'billed_units': {'search_units': 1}}}
    request = json.dumps({'query': 'two words', 'documents': ['three more words']})
    assert plugin.parse_responses([response], request=request) == (42, 0)
