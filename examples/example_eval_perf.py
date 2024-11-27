# Copyright (c) Alibaba, Inc. and its affiliates.
from evalscope.perf.main import run_perf_benchmark


def run_perf():
    task_cfg = {
        'url': 'http://127.0.0.1:8000/v1/chat/completions',
        'parallel': 1,
        'model': 'qwen2.5',
        'number': 15,
        'api': 'openai',
        'dataset': 'openqa',
        'debug': True,
    }
    run_perf_benchmark(task_cfg)


def run_perf_stream():
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


def run_perf_speed_benchmark():
    task_cfg = {
        'url': 'http://127.0.0.1:8000/v1/completions',
        'parallel': 1,
        'model': 'qwen2.5',
        'api': 'openai',
        'dataset': 'speed_benchmark',
        'debug': True,
    }
    run_perf_benchmark(task_cfg)


def run_perf_local():
    task_cfg = {
        'parallel': 1,
        'model': 'Qwen/Qwen2.5-0.5B-Instruct',
        'number': 5,
        'api': 'local',
        'dataset': 'openqa',
        'debug': True,
    }
    run_perf_benchmark(task_cfg)


def run_perf_local_stream():
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


def run_perf_local_speed_benchmark():
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


def run_perf_local_custom_prompt():
    task_cfg = {
        'parallel': 1,
        'model': 'Qwen/Qwen2.5-0.5B-Instruct',
        'api': 'local',
        'number': 10,
        'prompt': '写一个诗歌',
        'min_tokens': 100,
        'max_tokens': 1024,
        'debug': True,
    }
    run_perf_benchmark(task_cfg)


if __name__ == '__main__':
    run_perf_local_custom_prompt()
