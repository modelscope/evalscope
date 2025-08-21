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
                'max_tokens': 4096,
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

    def _run_dataset_load_test(self, dataset_name, dataset_args=None):
        """Helper method to test dataset loading."""

        self._run_dataset_test(dataset_name, dataset_args, use_mock=True, limit=None)

    # Math & Reasoning datasets
    def test_gsm8k(self):
        """Test GSM8K math reasoning dataset."""
        self._run_dataset_test('gsm8k')

    def test_gsm8k_local(self):
        """Test GSM8K math reasoning dataset with local path."""
        dataset_args = {
            'local_path': 'data/gsm8k',
        }
        self._run_dataset_test('gsm8k', dataset_args=dataset_args, use_mock=True)

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
            'few_shot_num': 2,
            'subset_list': ['computer science', 'math']
        }
        self._run_dataset_test('mmlu_pro', use_mock=False, dataset_args=dataset_args, repeats=2)

    def test_mmlu_redux(self):
        """Test MMLU-Redux reasoning dataset."""
        dataset_args = {
            'subset_list': ['abstract_algebra', 'computer_security'],
        }
        # self._run_dataset_load_test('mmlu_redux', dataset_args)
        self._run_dataset_test('mmlu_redux', dataset_args=dataset_args)

    def test_cmmlu(self):
        """Test C-MMLU reasoning dataset."""
        dataset_args = {
            'subset_list': ['agronomy', 'computer_security'],
            'few_shot_num': 0,
        }
        # self._run_dataset_load_test('cmmlu')
        self._run_dataset_test('cmmlu', dataset_args=dataset_args)

    def test_math_500(self):
        """Test MATH 500 dataset."""
        # self._run_dataset_load_test('math_500')
        self._run_dataset_test('math_500')

    def test_aime24(self):
        """Test AIME 2024 dataset."""
        self._run_dataset_test('aime24')

    def test_aime25(self):
        """Test AIME 2025 dataset."""
        self._run_dataset_test('aime25')

    def test_competition_math(self):
        """Test Competition Math dataset."""
        dataset_args = {
            'subset_list': ['Level 4']
        }
        self._run_dataset_test('competition_math', dataset_args)

    # Knowledge & QA datasets
    def test_arc(self):
        """Test ARC dataset."""
        # self._run_dataset_load_test('arc')
        dataset_args = {
            'subset_list': ['ARC-Easy', 'ARC-Challenge'],
            'few_shot_num': 2,
        }
        self._run_dataset_test('arc', dataset_args=dataset_args)

    def test_ceval(self):
        """Test CEval dataset."""
        dataset_args = {
            'subset_list': ['logic', 'law'],
            # 'few_shot_num': 0,
        }
        # self._run_dataset_load_test('ceval')
        self._run_dataset_test('ceval', dataset_args=dataset_args)

    def test_super_gpqa(self):
        """Test Super GPQA dataset."""
        # self._run_dataset_load_test('super_gpqa')

        dataset_args = {
            'subset_list': ['History', 'Psychology'],
            'few_shot_num': 0,
        }
        self._run_dataset_test('super_gpqa', dataset_args=dataset_args, ignore_errors=True)

    def test_gpqa(self):
        """Test GPQA dataset."""
        # self._run_dataset_load_test('gpqa_diamond')
        dataset_args = {
            'few_shot_num': 0,
        }
        self._run_dataset_test('gpqa_diamond', dataset_args=dataset_args, ignore_errors=True)

    def test_iquiz(self):
        """Test IQuiz dataset."""
        dataset_args = {
            'subset_list': ['IQ', 'EQ'],
            'few_shot_num': 0,
        }
        self._run_dataset_test('iquiz', dataset_args=dataset_args)

    def test_maritime_bench(self):
        """Test MaritimeBench dataset."""
        dataset_args = {
            'subset_list': ['default'],
            'few_shot_num': 0,
        }
        self._run_dataset_test('maritime_bench', dataset_args=dataset_args)

    def test_musr(self):
        """Test MuSR dataset."""
        dataset_args = {
            'subset_list': ['murder_mysteries', 'object_placements', 'team_allocation'],
            'few_shot_num': 0,
        }
        self._run_dataset_test('musr', dataset_args=dataset_args)

    def test_hellaswag(self):
        """Test HellaSwag dataset."""
        self._run_dataset_test('hellaswag')

    def test_truthful_qa(self):
        """Test TruthfulQA dataset."""
        dataset_args = {
            'extra_params': {
                'multiple_correct': True
            }
        }
        self._run_dataset_test('truthful_qa', dataset_args=dataset_args)

    def test_trivia_qa(self):
        """Test TriviaQA dataset."""
        self._run_dataset_test('trivia_qa')

    def test_race(self):
        """Test RACE dataset."""
        self._run_dataset_test('race')

    def test_winogrande(self):
        """Test winogrande"""
        self._run_dataset_test('winogrande')

    def test_bbh(self):
        dataset_args = {
            'subset_list': ['temporal_sequences', 'navigate'],
        }
        self._run_dataset_test('bbh', dataset_args=dataset_args)

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

    def test_general_mcq(self):
        """Test custom general MCQ dataset."""
        dataset_args = {
            'local_path': 'custom_eval/text/mcq',
            'subset_list': ['example']
        }
        self._run_dataset_test('general_mcq', dataset_args)

    def test_alpaca_eval(self):
        """Test AlpacaEval dataset."""
        self._run_dataset_test('alpaca_eval')

    def test_arena_hard(self):
        """Test Arena Hard dataset."""
        self._run_dataset_test('arena_hard', use_cache='outputs/20250818_211353')

    def test_frames(self):
        """Test Frames dataset."""
        dataset_args = {
            # 'local_path': '/root/.cache/modelscope/hub/datasets/iic/frames'
        }
        self._run_dataset_test('frames', dataset_args)

    def test_docmath(self):
        """Test DocMath dataset."""
        self._run_dataset_test('docmath')

    def test_drop(self):
        """Test DROP dataset."""
        dataset_args = {
            'few_shot_num': 3,
        }
        self._run_dataset_test('drop', dataset_args=dataset_args)

    def test_ifeval(self):
        """Test IFEval dataset."""
        self._run_dataset_test('ifeval')

    def test_needle_haystack(self):
        """Test Needle in Haystack dataset."""
        dataset_args = {
            'subset_list': ['english'],
            'extra_params': {
                'context_lengths_max': 10000,
                'context_lengths_num_intervals': 5,
                'document_depth_percent_intervals': 5,
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
            'extra_params': {
                'include_multi_modal': False
            }
        }
        self._run_dataset_test('hle', dataset_args)

    def test_process_bench(self):
        """Test ProcessBench dataset."""
        dataset_args = {
            'subset_list': ['gsm8k', 'math'],
        }
        self._run_dataset_test('process_bench', dataset_args, use_cache='outputs/20250819_161844')

    def test_humaneval(self):
        """Test HumanEval dataset."""
        dataset_args = {
            'metric_list': ['Pass@1', 'Pass@2', 'Pass@5']
        }
        self._run_dataset_test('humaneval', dataset_args, repeats=5)

    def test_live_code_bench(self):
        """Test LiveCodeBench dataset."""
        dataset_args = {
            'subset_list': ['v6'],
            'extra_params': {
                'start_date': '2024-08-01',
                'end_date': '2025-02-28'
            },
        }
        self._run_dataset_test('live_code_bench', dataset_args, judge_worker_num=1)

    def test_tool_bench(self):
        """Test ToolBench dataset."""
        self._run_dataset_test('tool_bench')

    def test_bfcl(self):
        """Test BFCL dataset."""
        dataset_args = {
            'subset_list': ['simple', 'live_multiple', 'multi_turn_base'],
            'extra_params': {
                'is_fc_model': True,
                'underscore_to_dot': True
            }
        }
        self._run_dataset_test('bfcl_v3', dataset_args)

    def test_tau_bench(self):
        dataset_args = {
            'extra_params': {
                'user_model': 'qwen-plus',
                'api_key': env.get('DASHSCOPE_API_KEY'),
                'api_base': 'https://dashscope.aliyuncs.com/compatible-mode/v1',
                'generation_config': {
                    'temperature': 0.7,
                    'max_new_tokens': 1024
                }
            }
        }
        self._run_dataset_test('tau_bench', dataset_args, limit=1)

if __name__ == '__main__':
    # Run specific test: python -m unittest test_eval.TestBenchmark.test_gsm8k
    # Run all tests: python -m unittest test_eval.TestBenchmark
    unittest.main()
