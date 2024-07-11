# Copyright (c) Alibaba, Inc. and its affiliates.

import os
import subprocess
import unittest

from llmuses.backend.opencompass import OpenCompassBackendManager
from llmuses.run import run_task
from llmuses.summarizer import Summarizer
from llmuses.utils import test_level_list, is_module_installed

from llmuses.utils.logger import get_logger

logger = get_logger(__name__)


class TestRunSwiftEval(unittest.TestCase):

    def setUp(self) -> None:
        logger.info(f'Init env for swift-eval UTs ...\n')

        self.model_name = 'llama3-8b-instruct'
        assert is_module_installed('llmuses'), 'Please install `llmuses` from pypi or source code.'

        if not is_module_installed('opencompass'):
            logger.warning('Note: ms-opencompass is not installed, installing it now...')
            subprocess.run('pip3 install ms-opencompass -U', shell=True, check=True)

        logger.info(f'>> Check swift installation: {is_module_installed("swift")}')
        if not is_module_installed('swift'):
            logger.warning('Note: modelscope swift is not installed, installing it now...')
            # subprocess.run('pip3 install ms-swift -U', shell=True, check=True)
            os.system('pip3 install ms-swift -U')

        if not is_module_installed('vllm'):
            logger.warning('Note: vllm is not installed, installing it now...\n')
            try:
                subprocess.run('pip3 install vllm -U', shell=True, check=True)
            except Exception as e:
                logger.warning(e)
                logger.warning(f'Failed to install vllm, use native swift deploy service instead.')

        logger.info(f'\nStaring run swift deploy ...')
        subprocess.run(f'swift deploy --model_type {self.model_name}', shell=True, check=True)

        self.all_datasets = OpenCompassBackendManager.list_datasets()
        assert len(self.all_datasets) > 0, f'Failed to list datasets from OpenCompass backend: {self.all_datasets}'

    def tearDown(self) -> None:
        # Stop the swift deploy model service
        logger.warning(f'\nStopping swift deploy ...')
        self.find_and_kill_service(self.model_name)

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

    @unittest.skipUnless(1 in test_level_list(), 'skip test in current test level')
    def test_run_task(self):
        # Prepare the config
        task_cfg = dict(
            eval_backend='OpenCompass',
            eval_config={'datasets': ['mmlu', 'ceval', 'ARC_c', 'gsm8k'],
                         'models': [
                             {'path': 'llama3-8b-instruct',
                              'openai_api_base': 'http://127.0.0.1:8000/v1/chat/completions',
                              'batch_size': 8},
                             # {'path': 'llama3-8b',
                             #  'is_chat': False,
                             #  'key': 'EMPTY',  # default to 'EMPTY', not available yet
                             #  'openai_api_base': 'http://127.0.0.1:8001/v1/completions'}
                         ],
                         'work_dir': 'outputs/llama3_eval_result',
                         'reuse': None,      # string, `latest` or timestamp, e.g. `20230516_144254`, default to None
                         'limit': '[2:5]',   # string or int or float, e.g. `[2:5]`, 5, 5.0, default to None, it means run all examples
                         },
        )

        # Submit the task
        logger.info(f'Start to run UT with cfg: {task_cfg}')
        run_task(task_cfg=task_cfg)

        # Get the final report with summarizer
        report_list = Summarizer.get_report_from_cfg(task_cfg)
        logger.info(f'>>The report list:\n{report_list}')

        assert len(report_list) > 0, f'Failed to get report list: {report_list}'


if __name__ == '__main__':
    unittest.main()
