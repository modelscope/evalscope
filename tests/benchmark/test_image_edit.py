# Copyright (c) Alibaba, Inc. and its affiliates.
from dotenv import dotenv_values

env = dotenv_values('.env')

import unittest

from evalscope.constants import EvalType, JudgeStrategy, ModelTask
from evalscope.utils.logger import get_logger
from tests.common import TestBenchmark

logger = get_logger()


class TestImageEditBenchmark(TestBenchmark):
    def setUp(self):
        """Setup common test configuration."""
        self.base_config = {
            'model': 'Qwen/Qwen-Image-Edit',
            'model_args':{
                'precision': 'bfloat16',
                'device_map': 'cuda:2'
            },
            'api_url': 'https://dashscope.aliyuncs.com/compatible-mode/v1',
            'api_key': env.get('DASHSCOPE_API_KEY'),
            'model_task': ModelTask.IMAGE_GENERATION,
            'eval_type': EvalType.IMAGE_EDITING,
            'eval_batch_size': 1,
            'limit': 5,
            'generation_config': {
                'true_cfg_scale': 4.0,
                'num_inference_steps': 50,
                'negative_prompt': ' ',
            },
            'judge_strategy': JudgeStrategy.AUTO,
            'judge_worker_num': 5,
            'judge_model_args': {
                'model_id': 'qwen2.5-vl-72b-instruct',
                'api_url': 'https://dashscope.aliyuncs.com/compatible-mode/v1',
                'api_key': env.get('DASHSCOPE_API_KEY'),
                'generation_config': {
                    'temperature': 0.0,
                    'max_tokens': 4096,
                }
            },
            'debug': True,
        }

    def test_gedit(self):
        """Test GEdit dataset."""
        dataset_args = {
            'extra_params':{
                'language': 'cn',
            }
        }
        self._run_dataset_test('gedit', dataset_args=dataset_args, use_cache='outputs/20250829_150058')

    def test_gedit_local(self):
        dataset_args = {
            'extra_params':{
                'language': 'cn',
                'local_file': 'outputs/example_edit.jsonl',
            }
        }
        self._run_dataset_test('gedit', dataset_args=dataset_args, model=None, model_id='offline_model')
