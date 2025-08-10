# Copyright (c) Alibaba, Inc. and its affiliates.
from dotenv import dotenv_values

env = dotenv_values('.env')

import unittest
from unittest import TestCase

from evalscope.config import TaskConfig
from evalscope.constants import EvalType, JudgeStrategy, OutputType
from evalscope.run import run_task
from evalscope.utils.logger import get_logger

logger = get_logger()


class TestBenchmark(TestCase):
    """Benchmark evaluation test cases."""

    def setUp(self):
        """Setup common test configuration."""
        self.base_config = {
            'model': 'qwen-plus',
            'api_url': 'https://dashscope.aliyuncs.com/compatible-mode/v1',
            'api_key': env.get('DASHSCOPE_API_KEY'),
            'eval_type': EvalType.SERVICE,
            'eval_batch_size': 5,
            'limit': 5,
            'generation_config': {
                'max_tokens': 2048,
                'temperature': 0.0,
                'seed': 42,
            },
            'judge_strategy': JudgeStrategy.AUTO,
            'judge_worker_num': 5,
            'judge_model_args': {
                'model_id': 'qwen2.5-72b-instruct',
                'api_url': 'https://dashscope.aliyuncs.com/compatible-mode/v1',
                'api_key': env.get('DASHSCOPE_API_KEY'),
                'generation_config': {
                    'temperature': 0.0,
                    'max_tokens': 4096
                }
            },
            'debug': True,
        }

    def _run_dataset_test(self, dataset_name, dataset_args=None, use_mock=False, **config_overrides):
        """Helper method to run test for a specific dataset."""
        config = self.base_config.copy()
        config['datasets'] = [dataset_name]

        if use_mock:
            config['eval_type'] = EvalType.MOCK_LLM

        # 应用配置覆盖
        config.update(config_overrides)

        if dataset_args:
            config['dataset_args'] = {dataset_name: dataset_args}

        task_cfg = TaskConfig(**config)
        run_task(task_cfg=task_cfg)

    # Math & Reasoning datasets
    def test_gsm8k(self):
        """Test GSM8K math reasoning dataset."""
        self._run_dataset_test('gsm8k')

    def test_mmlu(self):
        """Test MMLU reasoning dataset."""
        dataset_args = {
            'few_shot_num': 0,
            # 'subset_list': ['abstract_algebra', 'computer_security']
        }
        self._run_dataset_test('mmlu', use_mock=True, dataset_args=dataset_args)

    def test_mmlu_pro(self):
        """Test MMLU-Pro reasoning dataset."""
        dataset_args = {
            'few_shot_num': 0,
            'subset_list': ['computer science', 'math']
        }
        self._run_dataset_test('mmlu_pro', use_mock=False, dataset_args=dataset_args, repeats=2, use_cache='outputs/20250810_121607')

    def test_math_500(self):
        """Test MATH 500 dataset."""
        self._run_dataset_test('math_500')

    def test_aime24(self):
        """Test AIME 2024 dataset."""
        self._run_dataset_test('aime24')

    def test_competition_math(self):
        """Test Competition Math dataset."""
        dataset_args = {
            'subset_list': ['Level 4']
        }
        self._run_dataset_test('competition_math', dataset_args)

    # Knowledge & QA datasets
    def test_arc(self):
        """Test ARC dataset."""
        self._run_dataset_test('arc')

    def test_truthful_qa(self):
        """Test TruthfulQA dataset."""
        self._run_dataset_test('truthful_qa')

    def test_simple_qa(self):
        """Test SimpleQA dataset."""
        self._run_dataset_test('simple_qa')

    def test_chinese_simpleqa(self):
        """Test Chinese SimpleQA dataset."""
        dataset_args = {
            'subset_list': ['中华文化']
        }
        self._run_dataset_test('chinese_simpleqa', dataset_args)

    # Code datasets
    def test_live_code_bench(self):
        """Test LiveCodeBench dataset."""
        dataset_args = {
            'extra_params': {
                'start_date': '2024-08-01',
                'end_date': '2025-02-28'
            },
            'local_path': '/root/.cache/modelscope/hub/datasets/AI-ModelScope/code_generation_lite'
        }
        self._run_dataset_test('live_code_bench', dataset_args)

    def test_humaneval(self):
        """Test HumanEval dataset."""
        self._run_dataset_test('humaneval')

    # Custom & specialized datasets
    def test_general_qa(self):
        """Test custom general QA dataset."""
        dataset_args = {
            'local_path': 'custom_eval/text/qa',
            'subset_list': ['example']
        }
        self._run_dataset_test('general_qa', dataset_args)

    def test_alpaca_eval(self):
        """Test AlpacaEval dataset."""
        self._run_dataset_test('alpaca_eval')

    def test_arena_hard(self):
        """Test Arena Hard dataset."""
        self._run_dataset_test('arena_hard')

    def test_frames(self):
        """Test Frames dataset."""
        dataset_args = {
            'local_path': '/root/.cache/modelscope/hub/datasets/iic/frames'
        }
        self._run_dataset_test('frames', dataset_args)

    def test_docmath(self):
        """Test DocMath dataset."""
        self._run_dataset_test('docmath')

    def test_needle_haystack(self):
        """Test Needle in Haystack dataset."""
        dataset_args = {
            'subset_list': ['english'],
            'extra_params': {
                'show_score': True,
            }
        }
        self._run_dataset_test('needle_haystack', dataset_args)

    def test_ifeval(self):
        """Test IFEval dataset."""
        self._run_dataset_test('ifeval')

    def test_hle(self):
        """Test HLE dataset."""
        dataset_args = {
            'subset_list': ['Math', 'Other'],
        }
        self._run_dataset_test('hle', dataset_args)


if __name__ == '__main__':
    # Run specific test: python -m unittest test_eval.TestBenchmark.test_gsm8k
    # Run all tests: python -m unittest test_eval.TestBenchmark
    unittest.main()
