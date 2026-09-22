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


class TestMultiTurnPerTurnTokens(unittest.TestCase):
    """Unit tests for --multi-turn-per-turn-tokens CLI override."""

    def test_cli_override_applied_by_turn_index(self) -> None:
        args = Arguments(
            model='test-model',
            api='openai',
            multi_turn=True,
            number=2,
            parallel=1,
            max_tokens=2048,
            multi_turn_per_turn_tokens=[150, 150, 1000],
        )
        self.assertEqual(args.multi_turn_per_turn_tokens, [150, 150, 1000])

    def test_cli_json_string_parsed(self) -> None:
        args = Arguments(
            model='test-model',
            api='openai',
            multi_turn=True,
            number=2,
            parallel=1,
            multi_turn_per_turn_tokens='[100, null, 500]',
        )
        self.assertEqual(args.multi_turn_per_turn_tokens, [100, None, 500])

    def test_none_falls_back_to_default(self) -> None:
        args = Arguments(
            model='test-model',
            api='openai',
            multi_turn=True,
            number=2,
            parallel=1,
        )
        self.assertIsNone(args.multi_turn_per_turn_tokens)

    def test_shorter_list_leaves_later_turns_uncapped(self) -> None:
        args = Arguments(
            model='test-model',
            api='openai',
            multi_turn=True,
            number=2,
            parallel=1,
            multi_turn_per_turn_tokens=[100],
        )
        # Only turn 0 is overridden; turn 1+ falls back to Turn.max_tokens or global
        self.assertEqual(len(args.multi_turn_per_turn_tokens), 1)

    def test_invalid_type_rejected(self) -> None:
        with self.assertRaises(ValueError):
            Arguments(
                model='test-model',
                api='openai',
                multi_turn=True,
                number=2,
                parallel=1,
                multi_turn_per_turn_tokens='not-a-list',
            )

    def test_negative_value_rejected(self) -> None:
        with self.assertRaises(ValueError):
            Arguments(
                model='test-model',
                api='openai',
                multi_turn=True,
                number=2,
                parallel=1,
                multi_turn_per_turn_tokens=[100, -50],
            )


if __name__ == '__main__':
    unittest.main(buffer=False)
