# Copyright (c) Alibaba, Inc. and its affiliates.
from dotenv import dotenv_values

env = dotenv_values('.env')
import unittest

from evalscope.run import run_task
from evalscope.summarizer import Summarizer
from evalscope.utils import is_module_installed, test_level_list
from evalscope.utils.logger import get_logger

logger = get_logger()


class TestVLMEval(unittest.TestCase):

    def setUp(self) -> None:
        self._check_env('vlmeval')

    def tearDown(self) -> None:
        pass

    @staticmethod
    def _check_env(module_name: str):
        if is_module_installed(module_name):
            logger.info(f'{module_name} is installed.')
        else:
            raise ModuleNotFoundError(f'run: pip install {module_name}')

    @unittest.skipUnless(0 in test_level_list(), 'skip test in current test level')
    def test_run_vlm_eval_local(self):
        task_cfg = {
            'eval_backend': 'VLMEvalKit',
            'eval_config': {
                'data': ['SEEDBench_IMG', 'ChartQA_TEST'],
                'limit': 20,
                'mode': 'all',
                'model': [{
                    'name': 'qwen-vl-chat',
                    'model_path': '../models/Qwen-VL-Chat'
                }],  # model name for VLMEval config
                'nproc': 1,
                'reuse': True,
            },
            'work_dir': 'outputs',
            'use_cache': 'outputs/20241216_142838'
        }

        logger.info(f'>> Start to run task: {task_cfg}')

        run_task(task_cfg)

        logger.info('>> Start to get the report with summarizer ...')
        report_list = Summarizer.get_report_from_cfg(task_cfg)
        logger.info(f'\n>>The report list: {report_list}')

        assert len(report_list) > 0, f'Failed to get report list: {report_list}'


    @unittest.skipUnless(0 in test_level_list(), 'skip test in current test level')
    def test_run_vlm_api(self):
        task_cfg = {
            'eval_backend': 'VLMEvalKit',
            'eval_config': {
                'data': ['SEEDBench_IMG', 'ChartQA_TEST'],
                'limit': 5,
                'mode': 'all',
                'model': [
                    {'api_base': 'https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions',
                    'key': env.get('DASHSCOPE_API_KEY'),
                    'name': 'CustomAPIModel',
                    'temperature': 0.0,
                    'type': 'qwen2.5-vl-7b-instruct',
                    'img_size': -1,
                    'video_llm': False,
                    'max_tokens': 512,}
                    ],
                'nproc': 5,
                'reuse': False,
            },
            'work_dir': 'outputs',
            # 'use_cache': 'outputs/20241216_142838'
        }

        logger.info(f'>> Start to run task: {task_cfg}')

        run_task(task_cfg)

        logger.info('>> Start to get the report with summarizer ...')
        report_list = Summarizer.get_report_from_cfg(task_cfg)
        logger.info(f'\n>>The report list: {report_list}')

        assert len(report_list) > 0, f'Failed to get report list: {report_list}'

if __name__ == '__main__':
    unittest.main(buffer=False)
