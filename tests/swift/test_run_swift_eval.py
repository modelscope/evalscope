# Copyright (c) Alibaba, Inc. and its affiliates.

import os
import json
import time
import requests
import subprocess
import unittest

from llmuses.backend.opencompass import OpenCompassBackendManager
from llmuses.run import run_task
from llmuses.summarizer import Summarizer
from llmuses.utils import test_level_list, is_module_installed

from llmuses.utils.logger import get_logger

logger = get_logger(__name__)

DEFAULT_CHAT_MODEL_URL = 'http://127.0.0.1:8000/v1/chat/completions'
DEFAULT_BASE_MODEL_URL = 'http://127.0.0.1:8001/v1/completions'


class TestRunSwiftEval(unittest.TestCase):

    def setUp(self) -> None:
        logger.info(f'Init env for swift-eval UTs ...\n')

        self.model_name = 'llama3-8b-instruct'
        assert is_module_installed('llmuses'), 'Please install `llmuses` from pypi or source code.'

        logger.warning('Note: installing ms-opencompass ...')
        subprocess.run('pip3 install ms-opencompass -U', shell=True, check=True)

        logger.warning('Note: installing ms-swift ...')
        subprocess.run('pip3 install ms-swift -U', shell=True, check=True)

        logger.warning('vllm not installed, use native swift deploy service instead.')

        logger.info(f'\nStaring run swift deploy ...')
        self.process_swift_deploy = subprocess.Popen(f'swift deploy --model_type {self.model_name}',
                                                     text=True, shell=True,
                                                     stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        self.all_datasets = OpenCompassBackendManager.list_datasets()
        assert len(self.all_datasets) > 0, f'Failed to list datasets from OpenCompass backend: {self.all_datasets}'

    def tearDown(self) -> None:
        # Stop the swift deploy model service
        logger.warning(f'\nStopping swift deploy ...')
        self.process_swift_deploy.terminate()
        self.process_swift_deploy.wait()
        logger.info(f'Process swift-deploy terminated successfully.')

    @staticmethod
    def find_and_kill_pid(pids: list):
        if len(pids) > 0:
            for pid in pids:
                subprocess.run(["kill", str(pid)])
                logger.warning(f"Killed process {pid}.")
        else:
            logger.info(f"No pids found.")

    @staticmethod
    def find_and_kill_service(service_name):
        try:
            # find pid
            result = subprocess.run(
                ["ps", "-ef"], stdout=subprocess.PIPE, text=True
            )

            lines = result.stdout.splitlines()
            pids = []
            for line in lines:
                if service_name in line and "grep" not in line:
                    parts = line.split()
                    pid = parts[1]
                    pids.append(pid)

            if not pids:
                logger.info(f"No process found for {service_name}.")
            else:
                for pid in pids:
                    subprocess.run(["kill", pid])
                    logger.warning(f"Killed process {pid} for service {service_name}.")
        except Exception as e:
            logger.error(f"An error occurred: {e}")

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

    @unittest.skipUnless(1 in test_level_list(), 'skip test in current test level')
    def test_run_task(self):
        # Prepare the config
        task_cfg = dict(
            eval_backend='OpenCompass',
            eval_config={'datasets': ['mmlu', 'ceval', 'ARC_c', 'gsm8k'],
                         'models': [
                             {'path': 'llama3-8b-instruct',
                              'openai_api_base': DEFAULT_CHAT_MODEL_URL,
                              'batch_size': 8},
                         ],
                         'work_dir': 'outputs/llama3_eval_result',
                         'reuse': None,      # string, `latest` or timestamp, e.g. `20230516_144254`, default to None
                         'limit': '[2:5]',   # string or int or float, e.g. `[2:5]`, 5, 5.0, default to None, it means run all examples
                         },
        )

        # Check the service status
        data = {'model': self.model_name, 'messages': [{'role': 'user', 'content': 'who are you?'}]}
        assert self.check_service_status(DEFAULT_CHAT_MODEL_URL, data=data), f'Failed to check service status: {DEFAULT_CHAT_MODEL_URL}'

        # Submit the task
        logger.info(f'Start to run UT with cfg: {task_cfg}')
        run_task(task_cfg=task_cfg)

        # Get the final report with summarizer
        report_list = Summarizer.get_report_from_cfg(task_cfg)
        logger.info(f'>>The report list:\n{report_list}')

        assert len(report_list) > 0, f'Failed to get report list: {report_list}'


if __name__ == '__main__':
    unittest.main()
