# Copyright (c) Alibaba, Inc. and its affiliates.
import os
import subprocess
import torch
import unittest

from evalscope.constants import EvalType
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
        task_cfg = {'model': 'qwen/Qwen2-0.5B-Instruct',
                    'datasets': [
                        'mmlu_pro',
                        # 'bbh',
                        'hellaswag',
                        # 'gsm8k',
                        # 'arc'
                        # 'race',
                        # 'truthful_qa',
                        # 'trivia_qa',
                        ],
                    'limit': 20,
                    'debug': True}
        run_task(task_cfg=task_cfg)


    @unittest.skipUnless(0 in test_level_list(), 'skip test in current test level')
    def test_run_custom_task(self):
        from evalscope.config import TaskConfig

        task_cfg = TaskConfig(
            model='qwen/Qwen2-0.5B-Instruct',
            datasets=['ceval', 'general_qa'],  # 数据格式，选择题格式固定为 'ceval'
            dataset_args={
                'ceval': {
                    'local_path': 'custom_eval/text/mcq',  # 自定义数据集路径
                    'subset_list': [
                        'example'  # 评测数据集名称，上述 *_dev.csv 中的 *
                    ]
                },
                'general_qa': {
                    'local_path': 'custom_eval/text/qa',  # 自定义数据集路径
                    'subset_list': [
                        'example'  # 评测数据集名称，上述 *_dev.csv 中的 *
                    ]
                }
            },
        )
        run_task(task_cfg=task_cfg)

    @unittest.skipUnless(0 in test_level_list(), 'skip test in current test level')
    def test_run_humaneval(self):
        from evalscope.config import TaskConfig

        task_cfg = TaskConfig(
            model='qwen/Qwen2-0.5B-Instruct',
            datasets=['humaneval'],
            limit=2
        )

        run_task(task_cfg=task_cfg)

    @unittest.skipUnless(0 in test_level_list(), 'skip test in current test level')
    def test_run_server_model(self):
        from evalscope.config import TaskConfig

        task_cfg = TaskConfig(
            model='qwen2.5',
            api_url='http://127.0.0.1:8801/v1/chat/completions',
            api_key='EMPTY',
            eval_type=EvalType.SERVICE,
            datasets=[
                'mmlu_pro',
                # 'race',
                # 'trivia_qa',
                # 'cmmlu',
                # 'humaneval',
                # 'competition_math',
                # 'gsm8k',
                # 'arc',
                # 'ceval',
                # 'bbh',
                # 'hellaswag',
            ],
            limit=2,
            debug=True
        )

        run_task(task_cfg=task_cfg)


if __name__ == '__main__':
    unittest.main()
