# Copyright (c) Alibaba, Inc. and its affiliates.
"""Multi-turn conversation performance benchmark tests.

Covers random multi-turn, ShareGPT multi-turn, and SWE-Smith multi-turn
datasets.  In multi-turn mode ``--number`` is the total number of
conversations and ``--parallel`` is the number of concurrent conversations.
"""
import unittest

from evalscope.perf.arguments import Arguments
from evalscope.perf.main import run_perf_benchmark
from evalscope.perf.multi_turn_args import MultiTurnArgs
from tests.perf.perf_test_base import DASHSCOPE_CHAT_URL, LOCAL_CHAT_URL, PerfTestBase


class TestPerfMultiTurn(PerfTestBase):
    """Multi-turn conversation performance benchmarks."""

    def test_random_multi_turn(self):
        """Multi-turn benchmark with synthetic random conversations.

        Each conversation has 2-4 user turns.  ``number`` is the total turn
        budget (= total API requests), ``parallel`` is the concurrency.
        Requires a running chat/completions endpoint and a local tokenizer.
        """
        task_cfg = Arguments(
            parallel=[5, 10],
            number=[10, 20],
            model='Qwen2.5-0.5B-Instruct',
            url=LOCAL_CHAT_URL,
            api='openai',
            dataset='random_multi_turn',
            multi_turn=True,
            min_turns=2,
            max_turns=4,
            min_prompt_length=64,
            max_prompt_length=256,
            max_tokens=128,
            tokenizer_path='Qwen/Qwen2.5-0.5B-Instruct',
        )
        result = run_perf_benchmark(task_cfg)
        print(result)

    def test_share_gpt_zh_multi_turn(self):
        """Multi-turn benchmark with ShareGPT Chinese conversations.

        Uses the full user+assistant conversation from the dataset; assistant
        turns are replaced by real model outputs during the benchmark.
        Requires DASHSCOPE_API_KEY.
        """
        self.skip_without_api_key()

        task_cfg = Arguments(
            parallel=2,
            number=8,
            model='qwen-plus',
            url=DASHSCOPE_CHAT_URL,
            api_key=self.api_key,
            api='openai',
            dataset='share_gpt_zh_multi_turn',
            multi_turn=True,
            max_tokens=128,
            max_turns=4,
        )
        result = run_perf_benchmark(task_cfg)
        print(result)

    def test_swe_smith_multi_turn(self):
        """Multi-turn benchmark with SWE-Smith live construction.

        Uses the swe_smith dataset which constructs conversations on-the-fly
        with a large first-turn prompt (65000 chars) and shorter subsequent
        turns (500 chars).  Each conversation has exactly 12 turns.
        Requires DASHSCOPE_API_KEY.
        """
        self.skip_without_api_key()

        task_cfg = Arguments(
            parallel=4,
            number=8,
            model='qwen-plus',
            url=DASHSCOPE_CHAT_URL,
            api_key=self.api_key,
            api='openai',
            dataset='swe_smith',
            tokenizer_path='moonshotai/Kimi-K2.5',
            multi_turn=True,
            max_tokens=128,
            min_tokens=128,
            min_turns=12,
            max_turns=12,
            multi_turn_args=MultiTurnArgs(
                first_turn_length=65000,
                subsequent_turn_length=500,
                num_workers=4,
            ),
            seed=42,
            extra_args={'ignore_eos': True},
        )
        result = run_perf_benchmark(task_cfg)
        print(result)


if __name__ == '__main__':
    unittest.main(buffer=False)
