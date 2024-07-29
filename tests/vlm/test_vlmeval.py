# Copyright (c) Alibaba, Inc. and its affiliates.

import subprocess
import unittest
from evalscope.utils import test_level_list, is_module_installed
from evalscope.utils.logger import get_logger
from evalscope.run import run_task
from evalscope.summarizer import Summarizer

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
        task_cfg = {'eval_backend': 'VLMEvalKit',
                    'eval_config': {'LOCAL_LLM': 'models/Qwen2-7B-Instruct',
                                    'OPENAI_API_BASE': 'http://localhost:8866/v1/chat/completions',
                                    'OPENAI_API_KEY': 'EMPTY',
                                    'data': ['SEEDBench_IMG', 'ChartQA_TEST'],
                                    'limit': 20,
                                    'mode': 'all',
                                    'model': [{'model_path': '../models/internlm-xcomposer2d5-7b',   # path/to/model_dir
                                               'name': 'XComposer2d5'}],                 # model name for VLMEval config
                                    'nproc': 1,
                                    'rerun': True,
                                    'work_dir': 'output'}}
        
        logger.info(f'>> Start to run task: {task_cfg}')
        
        run_task(task_cfg)
        
        logger.info('>> Start to get the report with summarizer ...')
        report_list = Summarizer.get_report_from_cfg(task_cfg)
        logger.info(f'\n>>The report list: {report_list}')
        
    @unittest.skipUnless(0 in test_level_list(), 'skip test in current test level')
    def test_run_vlm_eval_api(self):
        task_cfg = {'eval_backend': 'VLMEvalKit',
                    'eval_config': {'LOCAL_LLM': 'models/Qwen2-7B-Instruct',                         # judge model id
                                    'OPENAI_API_BASE': 'http://localhost:8866/v1/chat/completions',  # judge model api
                                    'OPENAI_API_KEY': 'EMPTY',
                                    
                                    'data': ['SEEDBench_IMG', 'ChartQA_TEST'],
                                    'limit': 20,
                                    'mode': 'all',
                                    'model': [{'api_base': 'http://localhost:8000/v1/chat/completions',  # swfit deploy model api
                                               'key': 'EMPTY',
                                               'name': 'CustomAPIModel',                                # must be CustomAPIModel for swift
                                               'temperature': 0.0,
                                               'type': 'qwen-vl-chat'}],                                # swift model type
                                    'nproc': 1,
                                    'rerun': True,
                                    'work_dir': 'output'}}
        
        logger.info(f'>> Start to run task: {task_cfg}')
        
        run_task(task_cfg)
        
        logger.info('>> Start to get the report with summarizer ...')
        report_list = Summarizer.get_report_from_cfg(task_cfg)
        logger.info(f'\n>> The report list: {report_list}')
        
        
if __name__ == '__main__':
    unittest.main(buffer=False)