# Copyright (c) Alibaba, Inc. and its affiliates.
"""SLA auto-tuning performance benchmark tests.

Covers the two SLA auto-tune objectives: latency constraints (<=) and
throughput maximization (max TPS).  Both tests use a local endpoint.
"""
import unittest

from evalscope.perf.arguments import Arguments
from evalscope.perf.main import run_perf_benchmark
from tests.perf.perf_test_base import LOCAL_COMPLETIONS_URL, PerfTestBase


class TestPerfSLA(PerfTestBase):
    """SLA auto-tuning performance benchmarks."""

    def test_sla_auto_tune_latency_constraint(self):
        """SLA auto-tune with a latency constraint (p99 <= 8s).

        Starts at parallel=32 and binary-searches for the highest parallelism
        where p99 latency stays within 8 seconds.  Uses the random dataset
        with 1024-token prompts and 1024-token outputs.
        """
        task_cfg = Arguments(
            parallel=32,
            number=32,
            model='Qwen2.5-0.5B-Instruct',
            url=LOCAL_COMPLETIONS_URL,
            api='openai',
            dataset='random',
            min_tokens=1024,
            max_tokens=1024,
            max_prompt_length=1024,
            min_prompt_length=1024,
            tokenizer_path='Qwen/Qwen2.5-0.5B-Instruct',
            sla_auto_tune=True,
            sla_variable='parallel',
            sla_params=[{'p99_latency': '<=8'}],
            sla_num_runs=1,
            extra_args={'ignore_eos': True},
        )
        run_perf_benchmark(task_cfg)

    def test_sla_auto_tune_max_tps(self):
        """SLA auto-tune to maximize TPS (throughput).

        Starts at parallel=32 and binary-searches for the parallelism that
        yields the maximum tokens-per-second throughput.  Uses the random
        dataset with 1024-token prompts and 1024-token outputs.
        """
        task_cfg = Arguments(
            parallel=32,
            number=32,
            model='Qwen2.5-0.5B-Instruct',
            url=LOCAL_COMPLETIONS_URL,
            api='openai',
            dataset='random',
            min_tokens=1024,
            max_tokens=1024,
            max_prompt_length=1024,
            min_prompt_length=1024,
            tokenizer_path='Qwen/Qwen2.5-0.5B-Instruct',
            sla_auto_tune=True,
            sla_variable='parallel',
            sla_params=[{'tps': 'max'}],
            sla_num_runs=1,
            extra_args={'ignore_eos': True},
        )
        run_perf_benchmark(task_cfg)


if __name__ == '__main__':
    unittest.main(buffer=False)
