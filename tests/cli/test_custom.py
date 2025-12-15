# Copyright (c) Alibaba, Inc. and its affiliates.
from dotenv import dotenv_values

from tests.utils import test_level_list

env = dotenv_values('.env')

import os
import unittest

from evalscope.constants import EvalType, JudgeStrategy
from evalscope.utils.logger import get_logger
from tests.common import TestBenchmark

os.environ['EVALSCOPE_LOG_LEVEL'] = 'DEBUG'

logger = get_logger()


class TestRunCustom(TestBenchmark):
    def setUp(self):
        """Setup common test configuration."""
        self.base_config = {
            'api_url': 'https://dashscope.aliyuncs.com/compatible-mode/v1',
            'api_key': env.get('DASHSCOPE_API_KEY'),
            'eval_type': EvalType.SERVICE,
            'eval_batch_size': 10,
            'debug': True,
            'stream': True,
            'generation_config': {
                'temperature': 0,
                'n': 1,
                'max_tokens': 4096,
            },
            'ignore_errors': False,
            'judge_model_args': {
                'model_id': 'qwen-plus',
                'api_url': 'https://dashscope.aliyuncs.com/compatible-mode/v1',
                'api_key': env.get('DASHSCOPE_API_KEY'),
                'generation_config': {
                    'temperature': 0.0,
                    'max_tokens': 4096
                },
            },
            'judge_worker_num': 5,
            'judge_strategy': JudgeStrategy.LLM,
        }


    def test_run_custom_task_mcq(self):
        """Test custom MCQ dataset with local checkpoint."""
        self._run_dataset_test(
            dataset_name='general_mcq',
            dataset_args={
                'local_path': 'custom_eval/text/mcq',
                'subset_list': ['example'],
                'query_template': 'Question: {question}\n{choices}\nAnswer: {answer}'
            },
            model='Qwen/Qwen3-0.6B',
            model_args={'precision': 'torch.float16'},
            generation_config={
                'max_tokens': 512,
                'extra_body': {'enable_thinking': True},
            },
            eval_batch_size=5,
            limit=10,
        )


    def test_run_custom_task_qa(self):
        """Test custom QA dataset with local checkpoint."""
        self._run_dataset_test(
            dataset_name='general_qa',
            dataset_args={
                'local_path': 'custom_eval/text/qa',
                'subset_list': ['example']
            },
            model='Qwen/Qwen3-0.6B',
            model_args={'precision': 'torch.float16'},
            generation_config={
                'max_tokens': 512,
                'extra_body': {'enable_thinking': True},
            },
            eval_batch_size=5,
            limit=10,
        )


    def test_run_local_dataset(self):
        """Test trivia_qa with local dataset."""
        self._run_dataset_test(
            dataset_name='trivia_qa',
            dataset_args={
                'dataset_id': 'data/data/trivia_qa',
            },
            model='qwen-plus',
            limit=5,
        )


    def test_run_general_no_answer(self):
        """Test general_qa without reference answers using LLM judge."""
        self._run_dataset_test(
            dataset_name='general_qa',
            dataset_args={
                'dataset_id': 'custom_eval/text/qa',
                'subset_list': ['arena'],
            },
            model='qwen2.5-7b-instruct',
            limit=10,
        )

    def test_run_general_no_answer_with_judge(self):
        self._run_dataset_test(
            dataset_name='general_qa',
            dataset_args={
                'dataset_id': 'custom_eval/text/qa',
                'subset_list': ['arena'],
            },
            model='qwen2.5-7b-instruct',
            limit=10,
            judge_model_args={
                'model_id': 'qwen2.5-72b-instruct',
                'api_url': 'https://dashscope.aliyuncs.com/compatible-mode/v1',
                'api_key': env.get('DASHSCOPE_API_KEY'),
                'generation_config': {
                    'temperature': 0.0,
                    'max_tokens': 4096
                },
                'score_type': 'numeric',
            },
            judge_worker_num=5,
            judge_strategy=JudgeStrategy.LLM,
        )

    def test_run_general_with_answer(self):
        """Test general_qa with reference answers using LLM recall judge."""
        self._run_dataset_test(
            dataset_name='general_qa',
            dataset_args={
                'dataset_id': 'custom_eval/text/qa',
                'subset_list': ['example'],
            },
            model='qwen-plus',
            limit=10,
            judge_model_args={
                'model_id': 'qwen2.5-72b-instruct',
                'api_url': 'https://dashscope.aliyuncs.com/compatible-mode/v1',
                'api_key': env.get('DASHSCOPE_API_KEY'),
                'generation_config': {
                    'temperature': 0.0,
                    'max_tokens': 4096
                },
                'score_type': 'pattern',
            },
            judge_worker_num=1,
            judge_strategy=JudgeStrategy.LLM,
        )


    def test_run_general_arena(self):
        """Test general_arena for model comparison."""
        self._run_dataset_test(
            dataset_name='general_arena',
            dataset_args={
                'extra_params': {
                    'models': [
                        {
                            'name': 'qwen2.5-7b',
                            'report_path': 'outputs/20250819_165034/reports/qwen2.5-7b-instruct'
                        },
                        {
                            'name': 'qwen2.5-72b',
                            'report_path': 'outputs/20250819_164926/reports/qwen2.5-72b-instruct'
                        }
                    ],
                    'baseline': 'qwen2.5-72b'
                }
            },
            model_id='Arena',
            limit=10,
            judge_model_args={
                'model_id': 'qwen-plus',
                'api_url': 'https://dashscope.aliyuncs.com/compatible-mode/v1',
                'api_key': env.get('DASHSCOPE_API_KEY'),
                'generation_config': {
                    'temperature': 0.0,
                    'max_tokens': 8000
                },
            },
            judge_worker_num=5,
        )


    def test_run_general_vqa(self):
        """Test general_vqa adapter with OpenAI-compatible message format for multimodal QA."""
        self._run_dataset_test(
            dataset_name='general_vqa',
            dataset_args={
                'local_path': 'custom_eval/multimodal/vqa',
                'subset_list': ['example_openai'],
            },
            model='qwen-vl-plus',
            generation_config={'max_tokens': 512},
            eval_batch_size=2,
            limit=5,
        )


    def test_run_general_vmcq(self):
        """Test general_vmcq adapter with non-OpenAI MCQ format (MMMU-style)."""
        self._run_dataset_test(
            dataset_name='general_vmcq',
            dataset_args={
                'local_path': 'custom_eval/multimodal/mcq',
                'subset_list': ['example'],
            },
            model='qwen-vl-plus',
            generation_config={'max_tokens': 512},
            eval_batch_size=2,
            limit=5,
        )
