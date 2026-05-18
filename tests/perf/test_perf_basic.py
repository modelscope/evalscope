# Copyright (c) Alibaba, Inc. and its affiliates.
"""Basic performance benchmark tests.

Covers single-turn perf scenarios including openqa, random, speed_benchmark,
VL datasets, multi-parallel sweeps, sequential runs, ShareGPT, and warmup.
"""
from evalscope.perf.arguments import Arguments
from evalscope.perf.main import run_perf_benchmark
from evalscope.perf.utils.benchmark_util import Metrics
from tests.perf.perf_test_base import (
    DASHSCOPE_CHAT_URL,
    DASHSCOPE_COMPLETIONS_URL,
    LOCAL_CHAT_URL,
    LOCAL_COMPLETIONS_URL,
    PerfTestBase,
)


class TestPerfBasic(PerfTestBase):
    """Basic single-turn performance benchmarks."""

    # ------------------------------------------------------------------
    # OpenQA / simple chat
    # ------------------------------------------------------------------

    def test_basic_openqa(self):
        """Basic openqa dataset benchmark against a local endpoint.

        Uses the OpenAI-compatible chat/completions API with the openqa
        dataset.  Verifies that the benchmark runner completes without error
        for a small number of requests (15) at parallelism 1.
        """
        task_cfg = Arguments(
            url='http://127.0.0.1:8001/v1/chat/completions',
            parallel=1,
            model='qwen2.5',
            number=15,
            api='openai',
            dataset='openqa',
            debug=True,
        )
        run_perf_benchmark(task_cfg)

    def test_local_openqa(self):
        """Local model openqa benchmark (no remote API).

        Launches a local model via the ``local`` API backend and runs 5
        requests with the openqa dataset.  Useful for smoke-testing local
        inference without network dependencies.
        """
        task_cfg = Arguments(
            parallel=1,
            model='Qwen/Qwen2.5-0.5B-Instruct',
            number=5,
            api='local',
            dataset='openqa',
            debug=True,
        )
        run_perf_benchmark(task_cfg)

    # ------------------------------------------------------------------
    # Speed benchmark
    # ------------------------------------------------------------------

    def test_local_speed_benchmark(self):
        """Local model speed_benchmark dataset.

        Uses the ``local_vllm`` API backend with the speed_benchmark dataset,
        generating exactly 2048 tokens per request to measure raw throughput.
        """
        task_cfg = Arguments(
            parallel=1,
            model='Qwen/Qwen2.5-0.5B-Instruct',
            api='local_vllm',
            dataset='speed_benchmark',
            min_tokens=2048,
            max_tokens=2048,
            debug=True,
        )
        run_perf_benchmark(task_cfg)

    def test_remote_speed_benchmark(self):
        """DashScope speed_benchmark via the completions endpoint.

        Sends 8 requests with ``min_tokens=max_tokens=2048`` to the
        DashScope compatible-mode completions API.  Requires
        DASHSCOPE_API_KEY.
        """
        self.skip_without_api_key()

        task_cfg = Arguments(
            model='qwen2.5-coder-7b-instruct',
            url=DASHSCOPE_COMPLETIONS_URL,
            api_key=self.api_key,
            api='openai',
            number=8,
            dataset='speed_benchmark',
            min_tokens=2048,
            max_tokens=2048,
            seed=None,
            stream=False,
            extra_args={'ignore_eos': True},
        )
        run_perf_benchmark(task_cfg)

    # ------------------------------------------------------------------
    # Random dataset
    # ------------------------------------------------------------------

    def test_local_random(self):
        """Random prompt dataset against a local completions endpoint.

        Generates random prompts of 1024 tokens and requests 1024 output
        tokens per request.  Uses ``tokenize_prompt=True`` to avoid
        re-tokenization overhead.
        """
        task_cfg = Arguments(
            parallel=20,
            model='Qwen2.5-0.5B-Instruct',
            url=LOCAL_COMPLETIONS_URL,
            api='openai',
            dataset='random',
            min_tokens=1024,
            max_tokens=1024,
            prefix_length=0,
            min_prompt_length=1024,
            max_prompt_length=1024,
            number=20,
            tokenizer_path='Qwen/Qwen2.5-0.5B-Instruct',
            seed=None,
            tokenize_prompt=True,
            extra_args={'ignore_eos': True},
        )
        metrics_result, percentile_result = run_perf_benchmark(task_cfg)
        print(metrics_result)
        print(percentile_result)

    # ------------------------------------------------------------------
    # Completion endpoint / multi-parallel sweep
    # ------------------------------------------------------------------

    def test_completion_endpoint(self):
        """/v1/completions endpoint sweep with varying parallel & number.

        Runs two configurations: (parallel=1, number=2) and (parallel=2,
        number=4) using the DashScope completions API with random 1024-token
        prompts.  Requires DASHSCOPE_API_KEY.
        """
        self.skip_without_api_key()

        task_cfg = Arguments(
            parallel=[1, 2],
            number=[2, 4],
            model='qwen2.5-coder-7b-instruct',
            url=DASHSCOPE_COMPLETIONS_URL,
            api_key=self.api_key,
            api='openai',
            dataset='random',
            min_tokens=100,
            max_tokens=100,
            prefix_length=0,
            min_prompt_length=1024,
            max_prompt_length=1024,
            stream=True,
            tokenizer_path='Qwen/Qwen2.5-0.5B-Instruct',
            seed=None,
            extra_args={'ignore_eos': True},
        )
        metrics_result, percentile_result = run_perf_benchmark(task_cfg)
        print(metrics_result)
        print(percentile_result)

    def test_multi_parallel_sweep(self):
        """Multi-parallel sweep against DashScope chat/completions.

        Sweeps (parallel=1, number=2) and (parallel=2, number=4) with random
        1024-token prompts.  Requires DASHSCOPE_API_KEY.
        """
        self.skip_without_api_key()

        task_cfg = Arguments(
            parallel=[1, 2],
            number=[2, 4],
            warmup_num=3,
            model='qwen-plus',
            url=DASHSCOPE_CHAT_URL,
            api_key=self.api_key,
            api='openai',
            dataset='random',
            min_tokens=100,
            max_tokens=100,
            prefix_length=0,
            min_prompt_length=1024,
            max_prompt_length=1024,
            tokenizer_path='deepseek-ai/DeepSeek-R1-0528',
            seed=None,
            extra_args={'ignore_eos': True},
        )
        result = run_perf_benchmark(task_cfg)
        print(task_cfg.outputs_dir)
        print(result)

    # ------------------------------------------------------------------
    # Vision-Language (VL) datasets
    # ------------------------------------------------------------------

    def test_random_vl(self):
        """Random VL (vision-language) dataset sweep.

        Generates random image+text prompts with 2 images per request (512x512)
        and sweeps (parallel=1, number=2) and (parallel=2, number=4).
        Requires DASHSCOPE_API_KEY.
        """
        self.skip_without_api_key()

        task_cfg = Arguments(
            parallel=[1, 2],
            number=[2, 4],
            model='qwen-vl-max',
            url=DASHSCOPE_CHAT_URL,
            api_key=self.api_key,
            api='openai',
            dataset='random_vl',
            min_tokens=100,
            max_tokens=100,
            prefix_length=0,
            min_prompt_length=100,
            max_prompt_length=100,
            image_height=512,
            image_width=512,
            image_num=2,
            tokenizer_path='Qwen/Qwen2.5-VL-7B-Instruct',
            seed=None,
            extra_args={'ignore_eos': True},
        )
        metrics_result, percentile_result = run_perf_benchmark(task_cfg)
        print(metrics_result)
        print(percentile_result)

    def test_kontext_bench_vl(self):
        """KontextBench VL dataset sweep.

        Uses the kontext_bench dataset with the qwen-vl-max model.
        Sweeps (parallel=1, number=2) and (parallel=2, number=4).
        Requires DASHSCOPE_API_KEY.
        """
        self.skip_without_api_key()

        task_cfg = Arguments(
            parallel=[1, 2],
            number=[2, 4],
            model='qwen-vl-max',
            url=DASHSCOPE_CHAT_URL,
            api_key=self.api_key,
            api='openai',
            dataset='kontext_bench',
            min_tokens=100,
            max_tokens=100,
            tokenizer_path='Qwen/Qwen2.5-VL-7B-Instruct',
            seed=None,
            extra_args={'ignore_eos': True},
        )
        metrics_result, percentile_result = run_perf_benchmark(task_cfg)
        print(metrics_result)
        print(percentile_result)

    # ------------------------------------------------------------------
    # Sequential runs / ShareGPT
    # ------------------------------------------------------------------

    def test_sequential_runs(self):
        """Two sequential perf benchmark runs with the same config.

        Verifies that running ``run_perf_benchmark`` twice in succession does
        not leak state or raise errors.  Each run uses 1 request on the
        openqa dataset.  Requires DASHSCOPE_API_KEY.
        """
        self.skip_without_api_key()

        task_cfg1 = Arguments(
            parallel=1,
            number=1,
            model='qwen-plus',
            url=DASHSCOPE_CHAT_URL,
            api_key=self.api_key,
            api='openai',
            dataset='openqa',
            debug=True,
        )
        task_cfg2 = Arguments(
            parallel=1,
            number=1,
            model='qwen-plus',
            url=DASHSCOPE_CHAT_URL,
            api_key=self.api_key,
            api='openai',
            dataset='openqa',
            debug=True,
        )
        run_perf_benchmark(task_cfg1)
        run_perf_benchmark(task_cfg2)

    def test_share_gpt_zh(self):
        """ShareGPT Chinese conversation dataset sweep.

        Uses the share_gpt_zh dataset which contains real Chinese chat
        conversations.  Sweeps (parallel=1, number=2) and (parallel=2,
        number=4).  Requires DASHSCOPE_API_KEY.
        """
        self.skip_without_api_key()

        task_cfg = Arguments(
            parallel=[1, 2],
            number=[2, 4],
            model='qwen-plus',
            url=DASHSCOPE_CHAT_URL,
            api_key=self.api_key,
            api='openai',
            dataset='share_gpt_zh',
        )
        result = run_perf_benchmark(task_cfg)
        print(result)

    # ------------------------------------------------------------------
    # Warmup
    # ------------------------------------------------------------------

    def test_warmup_single_turn_absolute(self):
        """Single-turn warmup with absolute count (warmup_num=3).

        Sends 3 warmup requests followed by 10 benchmark requests.  Asserts
        that ``warmup_count`` resolves to 3 and that ``total_requests`` in
        the metrics equals 10 (warmup requests are excluded from metrics).
        Requires DASHSCOPE_API_KEY.
        """
        self.skip_without_api_key()

        task_cfg = Arguments(
            parallel=2,
            number=10,
            warmup_num=3,
            model='qwen-plus',
            url=DASHSCOPE_CHAT_URL,
            api_key=self.api_key,
            api='openai',
            dataset='openqa',
            max_tokens=64,
            stream=True,
        )
        self.assertEqual(task_cfg.warmup_count, 3)

        results = run_perf_benchmark(task_cfg)
        # Extract metrics from the single-run result dict
        run_key = list(results.keys())[0]
        metrics_result = results[run_key]['metrics']
        # Warmup requests must be excluded from metrics
        self.assertEqual(metrics_result.total_requests, 10)

    def test_warmup_single_turn_ratio(self):
        """Single-turn warmup with ratio mode (warmup_num=0.3).

        When warmup_num is between 0 and 1 it is treated as a ratio of
        ``number``.  0.3 * 10 = 3, so ``warmup_count`` should resolve to 3
        and ``total_requests`` should equal 10 (warmup excluded).
        Requires DASHSCOPE_API_KEY.
        """
        self.skip_without_api_key()

        task_cfg = Arguments(
            parallel=[2, 4],
            number=[5, 10],
            warmup_num=0.2,
            model='qwen-plus',
            url=DASHSCOPE_CHAT_URL,
            api_key=self.api_key,
            api='openai',
            dataset='openqa',
            max_tokens=64,
            stream=True,
        )
        results = run_perf_benchmark(task_cfg)

    def test_warmup_multi_turn_absolute(self):
        """Multi-turn warmup with absolute count (warmup_num=2).

        2 warmup conversations followed by 6 benchmark conversations, each
        with up to 3 turns.  Asserts ``warmup_count`` = 2 and that
        ``total_requests`` (individual turns) falls in [6, 18].
        Requires DASHSCOPE_API_KEY.
        """
        self.skip_without_api_key()

        task_cfg = Arguments(
            parallel=2,
            number=6,
            warmup_num=2,
            model='qwen-plus',
            url=DASHSCOPE_CHAT_URL,
            api_key=self.api_key,
            api='openai',
            dataset='share_gpt_zh_multi_turn',
            multi_turn=True,
            max_tokens=64,
            max_turns=3,
        )
        self.assertEqual(task_cfg.warmup_count, 2)

        results = run_perf_benchmark(task_cfg)
        run_key = list(results.keys())[0]
        metrics_result = results[run_key]['metrics']
        # In multi-turn mode total_requests counts individual turns,
        # not conversations.  6 benchmark conversations with max 3 turns each
        # should produce 6-18 turns; the 2 warmup conversations are excluded.
        self.assertGreaterEqual(metrics_result.total_requests, 6)
        self.assertLessEqual(metrics_result.total_requests, 18)
