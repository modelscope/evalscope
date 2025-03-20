# Copyright (c) Alibaba, Inc. and its affiliates.
import os
from dotenv import dotenv_values

env = dotenv_values('.env')
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import unittest

from evalscope.perf.main import run_perf_benchmark
from evalscope.utils import test_level_list


class TestPerf(unittest.TestCase):

    def setUp(self) -> None:
        pass

    def tearDown(self) -> None:
        pass

    @unittest.skipUnless(0 in test_level_list(), 'skip test in current test level')
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

    @unittest.skipUnless(0 in test_level_list(), 'skip test in current test level')
    def test_run_perf_stream(self):
        task_cfg = {
            'url': 'http://127.0.0.1:8000/v1/chat/completions',
            'parallel': 1,
            'model': 'qwen2.5',
            'number': 15,
            'api': 'openai',
            'dataset': 'openqa',
            'stream': True,
            'debug': True,
        }
        run_perf_benchmark(task_cfg)

    @unittest.skipUnless(0 in test_level_list(), 'skip test in current test level')
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

    @unittest.skipUnless(0 in test_level_list(), 'skip test in current test level')
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

    @unittest.skipUnless(0 in test_level_list(), 'skip test in current test level')
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

    @unittest.skipUnless(0 in test_level_list(), 'skip test in current test level')
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

    @unittest.skipUnless(0 in test_level_list(), 'skip test in current test level')
    def test_run_perf_local_random(self):
        from evalscope.perf.arguments import Arguments
        task_cfg = Arguments(
            parallel=20,
            model='Qwen2.5-0.5B-Instruct',
            url='http://127.0.0.1:8801/v1/chat/completions',
            api='openai',
            dataset='random',
            min_tokens=1024,
            max_tokens=1024,
            prefix_length=0,
            min_prompt_length=1024,
            max_prompt_length=1024,
            number=40,
            tokenizer_path='Qwen/Qwen2.5-0.5B-Instruct',
            seed=None,
            debug= True,
        )
        run_perf_benchmark(task_cfg)


if __name__ == '__main__':
    unittest.main(buffer=False)
