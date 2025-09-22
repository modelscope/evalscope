# Copyright (c) Alibaba, Inc. and its affiliates.
from dotenv import dotenv_values

env = dotenv_values('.env')

import unittest

from evalscope.constants import EvalType, JudgeStrategy, OutputType
from evalscope.utils.logger import get_logger
from tests.common import TestBenchmark

logger = get_logger()


class TestVLMBenchmark(TestBenchmark):
    """Benchmark evaluation test cases."""

    def setUp(self):
        """Setup common test configuration."""
        self.base_config = {
            'model': 'qwen-vl-plus',
            'api_url': 'https://dashscope.aliyuncs.com/compatible-mode/v1',
            'api_key': env.get('DASHSCOPE_API_KEY'),
            'eval_type': EvalType.SERVICE,
            'eval_batch_size': 5,
            'limit': 5,
            'generation_config': {
                'max_tokens': 2048,
                'temperature': 0.0,
                'seed': 42,
                'parallel_tool_calls': True
            },
            'judge_strategy': JudgeStrategy.AUTO,
            'judge_worker_num': 5,
            'judge_model_args': {
                'model_id': 'qwen2.5-72b-instruct',
                'api_url': 'https://dashscope.aliyuncs.com/compatible-mode/v1',
                'api_key': env.get('DASHSCOPE_API_KEY'),
                'generation_config': {
                    'temperature': 0.0,
                    'max_tokens': 4096,
                }
            },
            'debug': True,
        }

    def test_mmmu(self):
        dataset_args = {
            'subset_list':[
                'Accounting',
                'Agriculture',
                # 'Architecture_and_Engineering'
            ]
        }
        self._run_dataset_test('mmmu', dataset_args=dataset_args)

    def test_math_vista(self):
        dataset_args = {
            'subset_list': ['default']
        }
        self._run_dataset_test('math_vista', dataset_args=dataset_args)

    def test_mmmu_pro(self):
        dataset_args = {
            'subset_list':[
                'Accounting',
                # 'Agriculture',
            ],
            'extra_params': {
                'dataset_format': 'standard (4 options)',  # 'standard (4 options)', 'standard (10 options)', 'vision'
            },
        }
        self._run_dataset_test('mmmu_pro', dataset_args=dataset_args, limit=10)

    def test_qwen3_vl_collection(self):
        dataset_args = {
            'dataset_id': 'outputs/qwen3_vl_test.jsonl',
            'shuffle': True,
        }
        self._run_dataset_test('data_collection', dataset_args, limit=100)

    def test_real_world_qa(self):
        dataset_args = {
            'subset_list': ['default']
        }
        self._run_dataset_test('real_world_qa', dataset_args=dataset_args, limit=10)

    def test_ai2d(self):
        dataset_args = {
            'subset_list': ['default']
        }
        self._run_dataset_test('ai2d', dataset_args=dataset_args)

    def test_cc_bench(self):
        dataset_args = {
            'subset_list': ['cc']
        }
        self._run_dataset_test('cc_bench', dataset_args=dataset_args)

    def test_mm_bench(self):
        dataset_args = {
            'subset_list': ['cn', 'en']
        }
        self._run_dataset_test('mm_bench', dataset_args=dataset_args)

    def test_mm_star(self):
        dataset_args = {
            # 'subset_list': ['val']
        }
        self._run_dataset_test('mm_star', dataset_args=dataset_args)

    def test_omni_bench(self):
        dataset_args = {
            'extra_params': {
                'use_image': True, # Whether to use image input, if False, use text alternative image content.
                'use_audio': True, # Whether to use audio input, if False, use text alternative audio content.
            }
        }
        self._run_dataset_test('omni_bench', dataset_args=dataset_args, model='qwen-omni-turbo')

    def test_olympiad_bench(self):
        dataset_args = {
            'subset_list': [
                # 'OE_MM_maths_en_COMP',
                # 'OE_MM_maths_zh_CEE',
                # 'OE_MM_maths_zh_COMP',
                # 'OE_MM_physics_en_COMP',
                # 'OE_MM_physics_zh_CEE',
                # 'OE_TO_maths_en_COMP',
                # 'OE_TO_maths_zh_CEE',
                # 'OE_TO_maths_zh_COMP',
                # 'OE_TO_physics_en_COMP',
                # 'OE_TO_physics_zh_CEE',
                'TP_TO_maths_zh_CEE',
            ]
        }
        self._run_dataset_test('olympiad_bench', dataset_args=dataset_args)
