# Copyright (c) Alibaba, Inc. and its affiliates.
import os
import subprocess
import torch
import unittest

from evalscope.config import TaskConfig
from evalscope.constants import EvalType, JudgeStrategy, OutputType
from evalscope.run import run_task
from evalscope.utils import is_module_installed, test_level_list
from evalscope.utils.logger import get_logger

os.environ['LOG_LEVEL'] = 'DEBUG'

logger = get_logger()


class TestRun(unittest.TestCase):

    def setUp(self) -> None:
        logger.info('Init env for evalscope native run UTs ...\n')
        self._check_env('evalscope')

    def tearDown(self) -> None:
        pass

    @staticmethod
    def _check_env(module_name: str):
        if is_module_installed(module_name):
            logger.info(f'{module_name} is installed.')
        else:
            raise ModuleNotFoundError(f'run: pip install {module_name}')

    @unittest.skipUnless(0 in test_level_list(), 'skip test in current test level')
    def test_run_simple_eval(self):
        model = 'qwen/Qwen2-0.5B-Instruct'
        datasets = 'arc'  # arc ceval
        limit = 10

        cmd_simple = f'evalscope eval ' \
                     f'--model {model} ' \
                     f'--datasets {datasets} ' \
                     f'--limit {limit}'

        logger.info(f'Start to run command: {cmd_simple}')
        run_res = subprocess.run(cmd_simple, text=True, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        assert run_res.returncode == 0, f'Failed to run command: {cmd_simple}'
        logger.info(f'>>test_run_simple_eval stdout: {run_res.stdout}')
        logger.error(f'>>test_run_simple_eval stderr: {run_res.stderr}')

    @unittest.skipUnless(0 in test_level_list(), 'skip test in current test level')
    def test_run_eval_with_args(self):
        model = 'qwen/Qwen2-0.5B-Instruct'
        datasets = 'arc'  # arc ceval
        limit = 5
        dataset_args = '{"ceval": {"few_shot_num": 0, "few_shot_random": false}}'

        cmd_with_args = f'evalscope eval ' \
                        f'--model {model} ' \
                        f'--datasets {datasets} ' \
                        f'--limit {limit} ' \
                        f'--generation-config do_sample=false,temperature=0.0 ' \
                        f"""--dataset-args \'{dataset_args}\' """

        logger.info(f'Start to run command: {cmd_with_args}')
        run_res = subprocess.run(cmd_with_args, text=True, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        assert run_res.returncode == 0, f'Failed to run command: {cmd_with_args}'
        logger.info(f'>>test_run_eval_with_args stdout: {run_res.stdout}')
        logger.error(f'>>test_run_eval_with_args stderr: {run_res.stderr}')

    @unittest.skipUnless(0 in test_level_list(), 'skip test in current test level')
    def test_run_task(self):
        task_cfg = TaskConfig(
            model='qwen/Qwen2.5-0.5B-Instruct',
            datasets=[
                'iquiz',
                # 'ifeval',
                # 'mmlu',
                # 'mmlu_pro',
                # 'musr',
                # 'process_bench',
                # 'race',
                # 'trivia_qa',
                # 'cmmlu',
                # 'humaneval',
                # 'super_gpqa',
                # 'gsm8k',
                # 'bbh',
                # 'competition_math',
                # 'math_500',
                'aime24',
                'gpqa',
                # 'arc',
                # 'ceval',
                # 'hellaswag',
                # 'general_mcq',
                # 'general_qa'
            ],
            dataset_args={
                'mmlu': {
                    'subset_list': ['elementary_mathematics'],
                    'few_shot_num': 0
                },
                'mmlu_pro': {
                    'subset_list': ['math', 'health'],
                    'few_shot_num': 4
                },
                'ceval': {
                    'subset_list': [
                        'computer_network', 'operating_system', 'computer_architecture'
                    ],
                    'few_shot_num': 0
                },
                'cmmlu': {
                    'subset_list': ['elementary_chinese'],
                    'few_shot_num': 0
                },
                'bbh': {
                    'subset_list': ['word_sorting', 'movie_recommendation'],
                },
                'gpqa': {
                    'subset_list': ['gpqa_diamond'],
                    'few_shot_num': 0
                },
                'humaneval': {
                    'metric_list': ['Pass@1', 'Pass@2', 'Pass@5'],
                },
                'competition_math': {
                    'subset_list': ['Level 1']
                },
                'process_bench': {
                    'subset_list': ['gsm8k'],
                },
                'musr': {
                    'subset_list': ['murder_mysteries']
                },
                'general_mcq': {
                    'local_path': 'custom_eval/text/mcq',  # 自定义数据集路径
                    'subset_list': [
                        'example'  # 评测数据集名称，上述 *_dev.csv 中的 *
                    ],
                    'query_template': 'Question: {question}\n{choices}\nAnswer: {answer}'  # 问题模板
                },
                'general_qa': {
                    'local_path': 'custom_eval/text/qa',  # 自定义数据集路径
                    'subset_list': [
                        'example',  # 评测数据集名称，上述 *_dev.csv 中的 *
                        # 'test'
                    ],
                    'metric_list': ['AverageBLEU']
                },
                'super_gpqa': {
                    'subset_list': ['Philosophy', 'Education'],
                    'few_shot_num': 0
                },
                'ifeval': {
                    'filters': {
                        'remove_until': '</think>'
                    }
                }
            },
            limit=2,
            eval_batch_size=2,
            generation_config={
                'max_new_tokens': 2048,
                'temperature': 0.7,
                'num_return_sequences': 1,
            },
            # debug=True
        )
        run_task(task_cfg=task_cfg)


    @unittest.skipUnless(0 in test_level_list(), 'skip test in current test level')
    def test_run_custom_task(self):
        from evalscope.config import TaskConfig

        task_cfg = TaskConfig(
            model='qwen/Qwen2-0.5B-Instruct',
            datasets=['general_mcq', 'general_qa'],  # 数据格式，选择题格式固定为 'ceval'
            dataset_args={
                'general_mcq': {
                    'local_path': 'custom_eval/text/mcq',  # 自定义数据集路径
                    'subset_list': [
                        'example'  # 评测数据集名称，上述 *_dev.csv 中的 *
                    ],
                    'query_template': 'Question: {question}\n{choices}\nAnswer: {answer}'  # 问题模板
                },
                'general_qa': {
                    'local_path': 'custom_eval/text/qa',  # 自定义数据集路径
                    'subset_list': [
                        'example'  # 评测数据集名称，上述 *_dev.csv 中的 *
                    ]
                }
            },
        )
        res = run_task(task_cfg=task_cfg)
        print(res)

    @unittest.skipUnless(0 in test_level_list(), 'skip test in current test level')
    def test_run_humaneval(self):
        from evalscope.config import TaskConfig

        task_cfg = TaskConfig(
            model='qwen/Qwen2-0.5B-Instruct',
            datasets=[
                # 'math_500',
                # 'aime24',
                'competition_math'
            ],
            dataset_args={
                'competition_math': {
                    'subset_list': ['Level 4', 'Level 5']
                }
            },
            limit=5
        )

        run_task(task_cfg=task_cfg)

    @unittest.skipUnless(0 in test_level_list(), 'skip test in current test level')
    def test_run_server_model(self):
        from evalscope.config import TaskConfig

        task_cfg = TaskConfig(
            model='qwen2.5-7b-instruct',
            api_url='https://dashscope.aliyuncs.com/compatible-mode/v1',
            api_key= os.getenv('DASHSCOPE_API_KEY'),
            eval_type=EvalType.SERVICE,
            datasets=[
                # 'iquiz',
                # 'ifeval',
                # 'mmlu',
                # 'mmlu_pro',
                # 'musr',
                # 'process_bench',
                # 'race',
                # 'trivia_qa',
                # 'cmmlu',
                # 'humaneval',
                # 'gsm8k',
                # 'bbh',
                # 'competition_math',
                # 'math_500',
                # 'aime24',
                # 'gpqa',
                # 'arc',
                # 'ceval',
                'hellaswag',
                # 'general_mcq',
                # 'general_qa'
                # 'super_gpqa',
            ],
            dataset_args={
                'mmlu': {
                    'subset_list': ['elementary_mathematics', 'high_school_european_history', 'nutrition'],
                    'few_shot_num': 0
                },
                'mmlu_pro': {
                    'subset_list': ['math', 'health'],
                    'few_shot_num': 4
                },
                'ceval': {
                    'subset_list': [
                        'computer_network', 'operating_system', 'computer_architecture'
                    ],
                    'few_shot_num': 0
                },
                'cmmlu': {
                    'subset_list': ['elementary_chinese'],
                    'few_shot_num': 0
                },
                'bbh': {
                    'subset_list': ['word_sorting', 'movie_recommendation'],
                },
                'gpqa': {
                    # 'subset_list': ['gpqa_diamond'],
                    'few_shot_num': 0,
                    'local_path': './data/data/gpqa',
                },
                'humaneval': {
                    'metric_list': ['Pass@1', 'Pass@2', 'Pass@5'],
                },
                'competition_math': {
                    'subset_list': ['Level 1']
                },
                'process_bench': {
                    'subset_list': ['gsm8k'],
                },
                'musr': {
                    'subset_list': ['murder_mysteries']
                },
                'general_mcq': {
                    'local_path': 'custom_eval/text/mcq',  # 自定义数据集路径
                    'subset_list': [
                        'example'  # 评测数据集名称，上述 *_dev.csv 中的 *
                    ],
                    'query_template': 'Question: {question}\n{choices}\nAnswer: {answer}'  # 问题模板
                },
                'general_qa': {
                    'local_path': 'custom_eval/text/qa',  # 自定义数据集路径
                    'subset_list': [
                        'example',  # 评测数据集名称，上述 *_dev.csv 中的 *
                        # 'test'
                    ],
                    'metric_list': ['AverageBLEU']
                },
                'super_gpqa': {
                    # 'subset_list': ['Philosophy', 'Education'],
                    'few_shot_num': 0
                }
            },
            eval_batch_size=32,
            limit=15,
            # debug=True,
            stream=False,
            generation_config={
                'temperature': 0,
                'n': 1,
                'max_tokens': 4096,
            },
            # use_cache='./outputs/20250212_150525',
        )

        run_task(task_cfg=task_cfg)


    @unittest.skipUnless(0 in test_level_list(), 'skip test in current test level')
    def test_run_batch_eval(self):
        from evalscope.config import TaskConfig

        task_cfg = TaskConfig(
            model='LLM-Research/Llama-3.2-1B-Instruct',
            datasets=[
                # 'math_500',
                # 'aime24',
                # 'competition_math'
                # 'arc',
                'gsm8k'
                # 'truthful_qa'
            ],
            dataset_args={
                'competition_math': {
                    'subset_list': ['Level 4', 'Level 5']
                }
            },
            eval_batch_size=2,
            limit=5,
            generation_config={
                'max_new_tokens': 2048,
                'temperature': 0.7,
                'num_return_sequences': 2,
            }
        )

        run_task(task_cfg=task_cfg)

    @unittest.skipUnless(0 in test_level_list(), 'skip test in current test level')
    def test_run_judge_model(self):
        from evalscope.config import TaskConfig

        task_cfg = TaskConfig(
            model='qwen2.5-7b-instruct',
            api_url='https://dashscope.aliyuncs.com/compatible-mode/v1',
            api_key= os.getenv('DASHSCOPE_API_KEY'),
            eval_type=EvalType.SERVICE,
            datasets=[
                # 'math_500',
                # 'aime24',
                # 'competition_math',
                # 'arc',
                # 'gsm8k'
                # 'truthful_qa',
                # 'simple_qa',
                # 'chinese_simpleqa',
                'general_qa'
            ],
            dataset_args={
                'competition_math': {
                    'subset_list': ['Level 4']
                },
                'general_qa': {
                    'local_path': 'custom_eval/text/qa',  # 自定义数据集路径
                    'subset_list': [
                        'example',  # 评测数据集名称，上述 *_dev.csv 中的 *
                        # 'test'
                    ]
                },
            },
            eval_batch_size=5,
            limit=5,
            judge_strategy=JudgeStrategy.LLM,
            judge_model_args={
                'model_id': 'qwen2.5-7b-instruct',
                'api_url': 'https://dashscope.aliyuncs.com/compatible-mode/v1',
                'api_key': os.getenv('DASHSCOPE_API_KEY'),
            }
        )

        run_task(task_cfg=task_cfg)

if __name__ == '__main__':
    unittest.main()
