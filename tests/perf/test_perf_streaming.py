# Copyright (c) Alibaba, Inc. and its affiliates.
"""Streaming performance benchmark tests.

Covers SSE streaming against both the OpenAI-compatible chat/completions
endpoint and the local model backend.
"""
import json
import unittest

from evalscope.perf.arguments import Arguments
from evalscope.perf.main import run_perf_benchmark
from evalscope.perf.plugin.api.default_api import StreamedResponseHandler
from evalscope.perf.plugin.api.openai_responses_api import _extract_sse_data
from tests.perf.perf_test_base import LOCAL_CHAT_URL, PerfTestBase


class TestStreamedResponseHandler(unittest.TestCase):
    """Unit tests for SSE chunk parsing, covering multi-field SSE blocks
    (id:/event:/data:) and stream endings without a trailing separator."""

    def test_extracts_data_from_sse_event_with_metadata(self) -> None:
        handler = StreamedResponseHandler()

        messages = handler.add_chunk(b'id: chunk-1\nevent: message\ndata: {"choices": []}\n\n')

        self.assertEqual(messages, ['data: {"choices": []}'])
        self.assertEqual(json.loads(messages[0].removeprefix('data:').strip()), {'choices': []})

    def test_ignores_sse_metadata_without_data(self) -> None:
        handler = StreamedResponseHandler()

        messages = handler.add_chunk(b'event: ping\nid: keepalive\n\n')

        self.assertEqual(messages, [])

    def test_merges_multiple_data_lines_without_losing_content(self) -> None:
        handler = StreamedResponseHandler()

        messages = handler.add_chunk(b'event: message\ndata: {"a": 1,\ndata: "b": 2}\n\n')

        self.assertEqual(len(messages), 1)
        # The reconstructed message keeps a single leading 'data:' prefix, so
        # downstream per-line extractors (e.g. _extract_sse_data) must not
        # silently drop the continuation content.
        self.assertEqual(_extract_sse_data(messages[0]), messages[0].removeprefix('data:').strip())
        self.assertEqual(json.loads(messages[0].removeprefix('data:').strip()), {'a': 1, 'b': 2})

    def test_flushes_leftover_buffer_starting_with_metadata_without_trailing_separator(self) -> None:
        """A stream that ends right after the JSON payload (no trailing
        '\\n\\n') must still be parsed, even if the leftover buffer starts
        with SSE metadata fields (id:/event:) instead of 'data:'."""
        handler = StreamedResponseHandler()

        messages = handler.add_chunk(b'id: chunk-2\nevent: message\ndata: {"choices": []}')

        self.assertEqual(messages, ['data: {"choices": []}'])
        self.assertEqual(handler.buffer, '')

    def test_flushes_leftover_done_starting_with_metadata(self) -> None:
        handler = StreamedResponseHandler()

        messages = handler.add_chunk(b'event: done\ndata: [DONE]')

        self.assertEqual(messages, ['data: [DONE]'])
        self.assertEqual(handler.buffer, '')

    def test_waits_for_incomplete_json_buffer(self) -> None:
        handler = StreamedResponseHandler()

        self.assertEqual(handler.add_chunk(b'data: {"choices":'), [])
        self.assertEqual(handler.add_chunk(b' []}'), ['data: {"choices": []}'])


class TestPerfStreaming(PerfTestBase):
    """Streaming (SSE) performance benchmarks."""

    def test_stream_openai_chat(self):
        """OpenAI chat/completions streaming benchmark.

        Sends 15 streaming requests at parallelism 1 using the openqa
        dataset against a local OpenAI-compatible chat/completions endpoint.
        Verifies that the SSE stream is correctly consumed and metrics are
        collected.
        """
        task_cfg = Arguments(
            url=LOCAL_CHAT_URL,
            parallel=1,
            model='Qwen2.5-0.5B-Instruct',
            number=15,
            api='openai',
            dataset='openqa',
            stream=True,
            debug=True,
        )
        run_perf_benchmark(task_cfg)

    def test_stream_local_chat(self):
        """Local model streaming benchmark.

        Launches a local model via the ``local`` API backend with streaming
        enabled and runs 5 requests with the openqa dataset.  Verifies that
        the local inference engine streams tokens correctly.
        """
        task_cfg = Arguments(
            parallel=1,
            model='Qwen/Qwen2.5-0.5B-Instruct',
            number=5,
            api='local',
            dataset='openqa',
            stream=True,
            debug=True,
        )
        run_perf_benchmark(task_cfg)


if __name__ == '__main__':
    unittest.main(buffer=False)
