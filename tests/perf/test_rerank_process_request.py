# Copyright (c) Alibaba, Inc. and its affiliates.
"""Unit tests for ``OpenaiRerankPlugin.process_request``.

The rerank summary line (``top_score=...``) is built inside the same
``try`` block whose handler marks a request as failed, so a score that
cannot be formatted with ``:.4f`` used to turn a perfectly good HTTP 200
into a failed benchmark request.  The score is copied verbatim out of the
server's JSON and is never validated, so it can be ``null`` or a string.
"""
import unittest
from typing import Any, Dict
from unittest.mock import AsyncMock, MagicMock

from evalscope.perf.arguments import Arguments
from evalscope.perf.plugin.api.openai_rerank_api import OpenaiRerankPlugin


def _session_returning(payload: Dict[str, Any]) -> MagicMock:
    """Build a mock aiohttp session whose POST answers HTTP 200 with ``payload``."""
    response = MagicMock()
    response.status = 200
    response.json = AsyncMock(return_value=payload)
    response.__aenter__.return_value = response
    client_session = MagicMock()
    client_session.post.return_value = response
    return client_session


class TestRerankProcessRequest(unittest.IsolatedAsyncioTestCase):

    async def _run(self, payload: Dict[str, Any]) -> Any:
        plugin = OpenaiRerankPlugin(Arguments(model='test-rerank'))
        return await plugin.process_request(
            _session_returning(payload), 'http://localhost/v1/rerank', {}, {'query': 'q', 'documents': ['a']}
        )

    async def test_numeric_score_is_summarized(self) -> None:
        """Control case: a normal response keeps its four-decimal summary."""
        output = await self._run({
            'results': [{'index': 0, 'relevance_score': 0.8421}, {'index': 1, 'relevance_score': 0.1}],
            'usage': {'prompt_tokens': 21, 'total_tokens': 21},
        })

        self.assertTrue(output.success)
        self.assertEqual(output.generated_text, 'top_score=0.8421, num_results=2')
        self.assertEqual(output.prompt_tokens, 21)

    async def test_score_key_fallback_is_summarized(self) -> None:
        """Servers that name the field ``score`` keep the same summary."""
        output = await self._run({'results': [{'index': 0, 'score': 0.5}], 'usage': {'total_tokens': 12}})

        self.assertTrue(output.success)
        self.assertEqual(output.generated_text, 'top_score=0.5000, num_results=1')

    async def test_null_score_still_counts_as_a_successful_request(self) -> None:
        """``relevance_score: null`` must not fail an HTTP 200 request."""
        payload = {
            'results': [{'index': 3, 'relevance_score': None}, {'index': 0, 'relevance_score': 0.91}],
            'usage': {'prompt_tokens': 37, 'total_tokens': 37},
        }

        output = await self._run(payload)

        self.assertTrue(output.success)
        self.assertIsNone(output.error)
        self.assertEqual(output.generated_text, 'top_score=None, num_results=2')
        self.assertEqual(output.prompt_tokens, 37)
        self.assertEqual(output.response_messages, [payload])

    async def test_non_numeric_score_still_counts_as_a_successful_request(self) -> None:
        """A stringified score must not fail an HTTP 200 request either."""
        output = await self._run({'results': [{'index': 0, 'relevance_score': '0.9312'}]})

        self.assertTrue(output.success)
        self.assertIsNone(output.error)
        self.assertEqual(output.generated_text, 'top_score=0.9312, num_results=1')


if __name__ == '__main__':
    unittest.main()
