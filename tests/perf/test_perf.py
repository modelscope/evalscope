# Copyright (c) Alibaba, Inc. and its affiliates.
from dotenv import dotenv_values

env = dotenv_values('.env')
import tempfile
import unittest

from evalscope.perf.main import run_perf_benchmark
from evalscope.perf.sla.sla_run import SLAAutoTuner
from evalscope.perf.utils.benchmark_util import Metrics
from evalscope.perf.utils.db_util import PercentileMetrics
from tests.utils import test_level_list


class TestPerf(unittest.TestCase):

    def setUp(self) -> None:
        pass

    def tearDown(self) -> None:
        pass


    def test_run_perf(self):
        task_cfg = {
            'url': 'http://127.0.0.1:8001/v1/chat/completions',
            'parallel': 1,
            'model': 'qwen2.5',
            'number': 15,
            'api': 'openai',
            'dataset': 'openqa',
            # 'stream': True,
            'debug': True,
        }
        run_perf_benchmark(task_cfg)


    def test_run_perf_stream(self):
        task_cfg = {
            'url': 'http://127.0.0.1:8801/v1/chat/completions',
            'parallel': 1,
            'model': 'Qwen2.5-0.5B-Instruct',
            'number': 15,
            'api': 'openai',
            'dataset': 'openqa',
            'stream': True,
            'debug': True,
        }
        run_perf_benchmark(task_cfg)


    def test_run_perf_speed_benchmark(self):
        from evalscope.perf.arguments import Arguments
        task_cfg = Arguments(
            model='qwen2.5-coder-7b-instruct',
            url='https://dashscope.aliyuncs.com/compatible-mode/v1/completions',
            api_key=env.get('DASHSCOPE_API_KEY'),
            api='openai',
            number=8,
            dataset='speed_benchmark',
            min_tokens=2048,
            max_tokens=2048,
            seed=None,
            stream=False,
            extra_args={'ignore_eos': True}
        )

        run_perf_benchmark(task_cfg)


    def test_run_perf_local(self):
        task_cfg = {
            'parallel': 1,
            'model': 'Qwen/Qwen2.5-0.5B-Instruct',
            'number': 5,
            'api': 'local',
            'dataset': 'openqa',
            'debug': True,
        }
        run_perf_benchmark(task_cfg)


    def test_run_perf_local_stream(self):
        task_cfg = {
            'parallel': 1,
            'model': 'Qwen/Qwen2.5-0.5B-Instruct',
            'number': 5,
            'api': 'local',
            'dataset': 'openqa',
            'stream': True,
            'debug': True,
        }
        run_perf_benchmark(task_cfg)


    def test_run_perf_local_speed_benchmark(self):
        task_cfg = {
            'parallel': 1,
            'model': 'Qwen/Qwen2.5-0.5B-Instruct',
            'api': 'local_vllm',
            'dataset': 'speed_benchmark',
            'min_tokens': 2048,
            'max_tokens': 2048,
            'debug': True,
        }
        run_perf_benchmark(task_cfg)


    def test_run_perf_local_random(self):
        from evalscope.perf.arguments import Arguments
        task_cfg = Arguments(
            parallel=20,
            model='Qwen2.5-0.5B-Instruct',
            url='http://127.0.0.1:8801/v1/completions',
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
            extra_args={'ignore_eos': True}
        )
        metrics_result, percentile_result = run_perf_benchmark(task_cfg)
        print(metrics_result)
        print(percentile_result)

    def test_run_completion_endpoint(self):
        if not env.get('DASHSCOPE_API_KEY'):
            self.skipTest('DASHSCOPE_API_KEY is not set.')
            return

        from evalscope.perf.arguments import Arguments
        task_cfg = Arguments(
            parallel=[1, 2],
            number=[2, 4],
            model='qwen2.5-coder-7b-instruct',
            url='https://dashscope.aliyuncs.com/compatible-mode/v1/completions',
            api_key=env.get('DASHSCOPE_API_KEY'),
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
            extra_args={'ignore_eos': True}
        )
        metrics_result, percentile_result = run_perf_benchmark(task_cfg)
        print(metrics_result)
        print(percentile_result)


    def test_run_perf_multi_parallel(self):
        if not env.get('DASHSCOPE_API_KEY'):
            self.skipTest('DASHSCOPE_API_KEY is not set.')
            return

        from evalscope.perf.arguments import Arguments
        task_cfg = Arguments(
            parallel=[1, 2],
            number=[2, 4],
            model='deepseek-r1-0528',
            url='https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions',
            api_key=env.get('DASHSCOPE_API_KEY'),
            api='openai',
            dataset='random',
            min_tokens=100,
            max_tokens=100,
            prefix_length=0,
            min_prompt_length=1024,
            max_prompt_length=1024,
            tokenizer_path='deepseek-ai/DeepSeek-R1-0528',
            seed=None,
            extra_args={'ignore_eos': True}
        )
        result = run_perf_benchmark(task_cfg)
        print(task_cfg.outputs_dir)
        print(result)


    def test_run_perf_random_vl(self):
        from evalscope.perf.arguments import Arguments
        task_cfg = Arguments(
            parallel=[1, 2],
            number=[2, 4],
            model='qwen-vl-max',
            url='https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions',
            api_key=env.get('DASHSCOPE_API_KEY'),
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
            extra_args={'ignore_eos': True}
        )
        metrics_result, percentile_result = run_perf_benchmark(task_cfg)
        print(metrics_result)
        print(percentile_result)

    def test_run_perf_vl(self):
        from evalscope.perf.arguments import Arguments
        task_cfg = Arguments(
            parallel=[1, 2],
            number=[2, 4],
            model='qwen-vl-max',
            url='https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions',
            api_key=env.get('DASHSCOPE_API_KEY'),
            api='openai',
            dataset='kontext_bench',
            min_tokens=100,
            max_tokens=100,
            tokenizer_path='Qwen/Qwen2.5-VL-7B-Instruct',
            seed=None,
            extra_args={'ignore_eos': True}
        )
        metrics_result, percentile_result = run_perf_benchmark(task_cfg)
        print(metrics_result)
        print(percentile_result)

    def test_run_two_perfs(self):
        from evalscope.perf.arguments import Arguments
        task_cfg1 = Arguments(
            parallel=1,
            number=1,
            model='qwen-plus',
            url='https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions',
            api_key=env.get('DASHSCOPE_API_KEY'),
            api='openai',
            dataset='openqa',
            debug=True,
        )
        task_cfg2 = Arguments(
            parallel=1,
            number=1,
            model='qwen-plus',
            url='https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions',
            api_key=env.get('DASHSCOPE_API_KEY'),
            api='openai',
            dataset='openqa',
            debug=True,
        )
        run_perf_benchmark(task_cfg1)
        run_perf_benchmark(task_cfg2)

    def test_perf_visualizer_swanlab(self):
        from evalscope.perf.arguments import Arguments
        task_cfg = Arguments(
            parallel=1,
            number=2,
            model='qwen-plus',
            url='https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions',
            api_key=env.get('DASHSCOPE_API_KEY'),
            api='openai',
            dataset='openqa',
            min_tokens=512,
            max_tokens=512,
            stream=True,
            visualizer='swanlab',
            extra_args={'ignore_eos': True}
        )

        run_perf_benchmark(task_cfg)

    def test_perf_visualizer_clearml(self):
        from evalscope.perf.arguments import Arguments
        task_cfg = Arguments(
            parallel=[1, 2],
            number=[2, 4],
            model='qwen-plus',
            url='https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions',
            api_key=env.get('DASHSCOPE_API_KEY'),
            api='openai',
            dataset='openqa',
            min_tokens=512,
            max_tokens=512,
            stream=True,
            visualizer='clearml',
            extra_args={'ignore_eos': True}
        )

        run_perf_benchmark(task_cfg)

    def test_perf_sla_auto_tune_less_than(self):
        from evalscope.perf.arguments import Arguments
        task_cfg = Arguments(
            parallel=32,
            number=32,
            model='Qwen2.5-0.5B-Instruct',
            url='http://127.0.0.1:8801/v1/completions',
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
            extra_args={'ignore_eos': True}
        )

        run_perf_benchmark(task_cfg)

    def test_perf_sla_auto_tune_max(self):
        from evalscope.perf.arguments import Arguments
        task_cfg = Arguments(
            parallel=32,
            number=32,
            model='Qwen2.5-0.5B-Instruct',
            url='http://127.0.0.1:8801/v1/completions',
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
            extra_args={'ignore_eos': True}
        )

        run_perf_benchmark(task_cfg)

    def test_perf_embedding_random(self):
        from evalscope.perf.arguments import Arguments
        task_cfg = Arguments(
            parallel=[1, 2],
            number=[2, 4],
            model='text-embedding-v4',
            url='https://dashscope.aliyuncs.com/compatible-mode/v1/embeddings',
            api_key=env.get('DASHSCOPE_API_KEY'),
            api='openai_embedding',
            dataset='random_embedding',
            min_prompt_length=256,
            max_prompt_length=256,
            tokenizer_path='Qwen/Qwen3-Embedding-0.6B',
        )
        result = run_perf_benchmark(task_cfg)

    def test_perf_embedding(self):
        from evalscope.perf.arguments import Arguments
        task_cfg = Arguments(
            parallel=[1, 2],
            number=[2, 4],
            model='text-embedding-v4',
            url='https://dashscope.aliyuncs.com/compatible-mode/v1/embeddings',
            api_key=env.get('DASHSCOPE_API_KEY'),
            api='openai_embedding',
            dataset='embedding',
            tokenizer_path='Qwen/Qwen3-Embedding-0.6B',
            dataset_path='custom_eval/text/retrieval/queries.jsonl'
        )
        result = run_perf_benchmark(task_cfg)

    def test_perf_embedding_random_batch(self):
        from evalscope.perf.arguments import Arguments
        task_cfg = Arguments(
            parallel=[1, 2],
            number=[2, 4],
            model='text-embedding-v4',
            url='https://dashscope.aliyuncs.com/compatible-mode/v1/embeddings',
            api_key=env.get('DASHSCOPE_API_KEY'),
            api='openai_embedding',
            dataset='random_embedding_batch',
            min_prompt_length=256,
            max_prompt_length=256,
            tokenizer_path='Qwen/Qwen3-Embedding-0.6B',
            extra_args={'batch_size': 8}
        )
        result = run_perf_benchmark(task_cfg)

    def test_perf_embedding_batch(self):
        from evalscope.perf.arguments import Arguments
        task_cfg = Arguments(
            parallel=[1, 2],
            number=[2, 4],
            model='text-embedding-v4',
            url='https://dashscope.aliyuncs.com/compatible-mode/v1/embeddings',
            api_key=env.get('DASHSCOPE_API_KEY'),
            api='openai_embedding',
            dataset='embedding_batch',
            tokenizer_path='Qwen/Qwen3-Embedding-0.6B',
            dataset_path='custom_eval/text/retrieval/queries.jsonl'
        )
        result = run_perf_benchmark(task_cfg)

    def test_perf_rerank_random(self):
        from evalscope.perf.arguments import Arguments
        task_cfg = Arguments(
            parallel=[1, 2],
            number=[1000, 1000],
            model='qwen3-rerank',
            url='https://dashscope.aliyuncs.com/compatible-api/v1/reranks',
            api_key=env.get('DASHSCOPE_API_KEY'),
            api='openai_rerank',
            dataset='random_rerank',
            min_prompt_length=256,
            max_prompt_length=256,
            tokenizer_path='Qwen/Qwen3-Embedding-0.6B',
            extra_args={
                'num_documents': 5,
                'document_length_ratio': 3,
            }
        )
        result = run_perf_benchmark(task_cfg)

    def test_perf_rerank(self):
        from evalscope.perf.arguments import Arguments
        task_cfg = Arguments(
            parallel=[1, 2],
            number=[2, 4],
            model='qwen3-rerank',
            url='https://dashscope.aliyuncs.com/compatible-api/v1/reranks',
            api_key=env.get('DASHSCOPE_API_KEY'),
            api='openai_rerank',
            dataset='rerank',
            tokenizer_path='Qwen/Qwen3-Embedding-0.6B',
            dataset_path='custom_eval/text/rerank/example.jsonl'
        )
        result = run_perf_benchmark(task_cfg)

    def test_perf_share_gpt(self):
        from evalscope.perf.arguments import Arguments
        task_cfg = Arguments(
            parallel=[1, 2],
            number=[2, 4],
            model='qwen-plus',
            url='https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions',
            api_key=env.get('DASHSCOPE_API_KEY'),
            api='openai',
            dataset='share_gpt_zh',
        )
        result = run_perf_benchmark(task_cfg)

    def test_run_perf_multi_turn_random(self):
        """Multi-turn benchmark with synthetic random conversations.

        Each conversation has 2-4 user turns.  ``--number`` is the total turn
        budget (= total API requests), ``--parallel`` is the concurrency.
        Requires a running chat/completions endpoint and a local tokenizer.
        """
        from evalscope.perf.arguments import Arguments
        task_cfg = Arguments(
            parallel=[5, 10],
            number=[10, 20],
            model='Qwen2.5-0.5B-Instruct',
            url='http://127.0.0.1:8801/v1/chat/completions',
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

    def test_run_perf_multi_turn_share_gpt(self):
        """Multi-turn benchmark with ShareGPT Chinese conversations.

        Uses the full user+assistant conversation from the dataset; assistant
        turns are replaced by real model outputs during the benchmark.
        Requires DASHSCOPE_API_KEY to be set in .env.
        """
        if not env.get('DASHSCOPE_API_KEY'):
            self.skipTest('DASHSCOPE_API_KEY is not set.')
            return

        from evalscope.perf.arguments import Arguments
        task_cfg = Arguments(
            parallel=2,
            number=5,
            model='qwen-plus',
            url='https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions',
            api_key=env.get('DASHSCOPE_API_KEY'),
            api='openai',
            dataset='share_gpt_zh_multi_turn',
            multi_turn=True,
            max_turns=4,
        )
        result = run_perf_benchmark(task_cfg)
        print(result)

    def test_run_perf_multi_turn_swe_smith(self):
        from evalscope.perf.arguments import Arguments
        from evalscope.perf.multi_turn_args import MultiTurnArgs
        task_cfg = Arguments(
            parallel=[5, 10],
            number=[10, 20],
            model='Qwen2.5-0.5B-Instruct',
            url='http://127.0.0.1:8801/v1/chat/completions',
            api='openai',
            dataset='swe_smith',
            tokenizer_path='Qwen/Qwen2.5-0.5B-Instruct',
            multi_turn=True,
            max_tokens=128,
            min_tokens=128,
            multi_turn_args=MultiTurnArgs(
                min_turns=2,
                max_turns=4,
                first_turn_length=8192,
                subsequent_turn_length=1024,
                max_context_length=12000
            ),
            seed=42,
            extra_args={'ignore_eos': True}
        )
        result = run_perf_benchmark(task_cfg)
        print(result)

if __name__ == '__main__':
    unittest.main(buffer=False)
