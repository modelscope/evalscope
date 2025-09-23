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
        task_cfg = {
            'url': 'http://127.0.0.1:8001/v1/completions',
            'parallel': 1,
            'model': 'qwen2.5',
            'api': 'openai',
            'dataset': 'speed_benchmark',
            'min_tokens': 2048,
            'max_tokens': 2048,
            'debug': True,
        }
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
            stream=False,
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
        metrics_result, percentile_result = run_perf_benchmark(task_cfg)
        print(metrics_result)
        print(percentile_result)


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

if __name__ == '__main__':
    unittest.main(buffer=False)
