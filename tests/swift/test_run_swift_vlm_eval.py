# Copyright (c) Alibaba, Inc. and its affiliates.
import json
import time
import requests
import os
import subprocess
import unittest
import shutil
from evalscope.backend.vlm_eval_kit import VLMEvalKitBackendManager
from evalscope.utils import test_level_list, is_module_installed
from evalscope.utils.logger import get_logger
from evalscope.run import run_task
from evalscope.summarizer import Summarizer

logger = get_logger(__name__)

DEFAULT_CHAT_MODEL_URL = 'http://127.0.0.1:8000/v1/chat/completions'
DEFAULT_API_KEY = 'EMPTY'
DEFAULT_MODEL_NAME = 'CustomAPIModel'
DEFAULT_WORK_DIR = 'outputs'

class TestRunSwiftVLMEval(unittest.TestCase):

    def setUp(self) -> None:
        logger.info(f'Init env for swift-eval UTs ...\n')
        assert is_module_installed('llmuses'), 'Please install `llmuses` from pypi or source code.'
        

        logger.warning('Note: installing ms-vlmeval ...')
        subprocess.run('pip3 install ms-vlmeval -U', shell=True, check=True)

        logger.warning('Note: installing ms-swift ...')
        subprocess.run('pip3 install ms-swift -U', shell=True, check=True)
        
        if os.path.exists(DEFAULT_WORK_DIR):
            shutil.rmtree(DEFAULT_WORK_DIR)
            logger.info(f'Removed work dir: {os.path.abspath(DEFAULT_WORK_DIR)} \n')
        
        logger.info(f'\nStaring run swift deploy ...')
        self.model_name = 'qwen-vl-chat'
        self.process_swift_deploy = subprocess.Popen(f'swift deploy --model_type {self.model_name} --infer_backend pt',
                                                     text=True, shell=True,
                                                     stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        self.all_datasets = VLMEvalKitBackendManager.list_supported_datasets()
        assert len(self.all_datasets) > 0, f'Failed to list datasets from VLMEvalKit backend: {self.all_datasets}'

    def tearDown(self) -> None:
        # Stop the swift deploy model service
        logger.warning(f'\nStopping swift deploy ...')
        self.process_swift_deploy.kill()
        self.process_swift_deploy.wait()
        logger.info(f'Process swift-deploy terminated successfully.')

    @staticmethod
    def _check_env(module_name: str):
        if is_module_installed(module_name):
            logger.info(f'{module_name} is installed.')
        else:
            raise ModuleNotFoundError(f'run: pip install {module_name}')

    @staticmethod
    def check_service_status(url: str, data: dict, retries: int = 20, delay: int = 10):
        for i in range(retries):
            try:
                logger.info(f"Attempt {i + 1}: Checking service at {url} ...")
                response = requests.post(url,
                                         data=json.dumps(data),
                                         headers={'Content-Type': 'application/json'},
                                         timeout=30)
                if response.status_code == 200:
                    logger.info(f"Service at {url} is available !\n\n")
                    return True
                else:
                    logger.info(f"Service at {url} returned status code {response.status_code}.")
            except requests.exceptions.RequestException as e:
                logger.info(f"Attempt {i + 1}: An error occurred: {e}")

            time.sleep(delay)

        logger.info(f"Service at {url} is not available after {retries} retries.")
        return False
        
    @unittest.skipUnless(0 in test_level_list(), 'skip test in current test level')
    def test_run_api(self):
        api_base = DEFAULT_CHAT_MODEL_URL
        task_cfg = {'eval_backend': 'VLMEvalKit',
                    'eval_config': {
                            'data': ['SEEDBench_IMG', 'ChartQA_TEST'],
                            'limit': 30,
                            'mode': 'all',
                            'model': [{'api_base': api_base,  # swfit deploy model api
                                        'key': DEFAULT_API_KEY,
                                        'name': DEFAULT_MODEL_NAME,         # must be CustomAPIModel for swift
                                        'temperature': 0.0,
                                        'type': self.model_name}],          # swift model type
                            'nproc': 1,
                            'rerun': True,
                            'work_dir': DEFAULT_WORK_DIR
                            }
                    }
        
        # Check the service status
        data = {'model': self.model_name, 'messages': [{'role': 'user', 'content': 'who are you?'}]}
        assert self.check_service_status(api_base, data=data), f'Failed to check service status: {api_base}'
        
        logger.info(f'>> Start to run task: {task_cfg}')
        
        run_task(task_cfg)
        
        logger.info('>> Start to get the report with summarizer ...')
        report_list = Summarizer.get_report_from_cfg(task_cfg)
        logger.info(f'\n>> The report list: {report_list}')
        
        assert len(report_list) > 0, f'Failed to get report list: {report_list}'
    

        
        
if __name__ == '__main__':
    unittest.main(buffer=False)