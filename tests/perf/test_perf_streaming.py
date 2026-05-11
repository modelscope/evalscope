# Copyright (c) Alibaba, Inc. and its affiliates.
"""Streaming performance benchmark tests.

Covers SSE streaming against both the OpenAI-compatible chat/completions
endpoint and the local model backend.
"""
import unittest

from evalscope.perf.arguments import Arguments
from evalscope.perf.main import run_perf_benchmark
from tests.perf.perf_test_base import LOCAL_CHAT_URL, PerfTestBase


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
