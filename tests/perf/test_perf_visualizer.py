# Copyright (c) Alibaba, Inc. and its affiliates.
"""Visualizer integration performance benchmark tests.

Covers SwanLab and ClearML visualizer backends.  Both tests stream 512-token
responses and log metrics to the respective visualizer.  Requires
DASHSCOPE_API_KEY.
"""
import unittest

from evalscope.perf.arguments import Arguments
from evalscope.perf.main import run_perf_benchmark
from tests.perf.perf_test_base import DASHSCOPE_CHAT_URL, PerfTestBase


class TestPerfVisualizer(PerfTestBase):
    """Visualizer integration performance benchmarks."""

    def test_visualizer_swanlab(self):
        """SwanLab visualizer integration benchmark.

        Runs a single-parallelism streaming benchmark with 2 requests and
        logs all metrics to SwanLab.  Each request generates up to 512
        tokens with ``ignore_eos`` to avoid early termination.
        Requires DASHSCOPE_API_KEY.
        """
        self.skip_without_api_key()

        task_cfg = Arguments(
            parallel=1,
            number=2,
            model='qwen-plus',
            url=DASHSCOPE_CHAT_URL,
            api_key=self.api_key,
            api='openai',
            dataset='openqa',
            min_tokens=512,
            max_tokens=512,
            stream=True,
            visualizer='swanlab',
            extra_args={'ignore_eos': True},
        )
        run_perf_benchmark(task_cfg)

    def test_visualizer_clearml(self):
        """ClearML visualizer integration benchmark sweep.

        Sweeps (parallel=1, number=2) and (parallel=2, number=4) with
        streaming enabled, logging all metrics to ClearML.  Each request
        generates up to 512 tokens with ``ignore_eos``.
        Requires DASHSCOPE_API_KEY.
        """
        self.skip_without_api_key()

        task_cfg = Arguments(
            parallel=[1, 2],
            number=[2, 4],
            model='qwen-plus',
            url=DASHSCOPE_CHAT_URL,
            api_key=self.api_key,
            api='openai',
            dataset='openqa',
            min_tokens=512,
            max_tokens=512,
            stream=True,
            visualizer='clearml',
            extra_args={'ignore_eos': True},
        )
        run_perf_benchmark(task_cfg)


if __name__ == '__main__':
    unittest.main(buffer=False)
