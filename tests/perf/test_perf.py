# Copyright (c) Alibaba, Inc. and its affiliates.
from dotenv import dotenv_values

env = dotenv_values('.env')
import unittest

from evalscope.perf.main import run_perf_benchmark
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
            model='Qwen3-1.7B',
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
            model='qwen-plus',
            url='https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions',
            api_key=env.get('DASHSCOPE_API_KEY'),
            api='openai',
            dataset='random',
            min_tokens=100,
            max_tokens=100,
            prefix_length=0,
            min_prompt_length=1024,
            max_prompt_length=1024,
            tokenizer_path='Qwen/Qwen2.5-0.5B-Instruct',
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

if __name__ == '__main__':
    unittest.main(buffer=False)
