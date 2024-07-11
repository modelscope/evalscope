# Copyright (c) Alibaba, Inc. and its affiliates.

import unittest

from llmuses.backend.opencompass import OpenCompassBackendManager
from llmuses.run import run_task
from llmuses.summarizer import Summarizer
from llmuses.utils import test_level_list, is_module_installed

from llmuses.utils.logger import get_logger

logger = get_logger()


class TestRunSwiftEval(unittest.TestCase):

    def setUp(self) -> None:
        assert is_module_installed('llmuses'), 'run: pip install llmuses -U'
        assert is_module_installed('ms-opencompass'), 'run: pip install ms-opencompass -U'

        self.all_datasets = OpenCompassBackendManager.list_datasets()
        assert len(self.all_datasets) > 0, f'Failed to list datasets from OpenCompass backend: {self.all_datasets}'

    def tearDown(self) -> None:
        pass

    @unittest.skipUnless(0 in test_level_list(), 'skip test in current test level')
    def test_run_task(self):
        # Prepare the config
        task_cfg = dict(
            eval_backend='OpenCompass',
            eval_config={'datasets': ['mmlu', 'ceval', 'ARC_c', 'gsm8k'],
                         'models': [
                             {'path': 'llama3-8b-instruct',
                              'openai_api_base': 'http://127.0.0.1:8000/v1/chat/completions',
                              'batch_size': 8},
                             {'path': 'llama3-8b',
                              'is_chat': False,
                              'key': 'EMPTY',  # default to 'EMPTY', not available yet
                              'openai_api_base': 'http://127.0.0.1:8001/v1/completions'}
                         ],
                         'work_dir': 'outputs/llama3_eval_result',
                         'reuse': None,      # string, `latest` or timestamp, e.g. `20230516_144254`, default to None
                         'limit': '[2:5]',   # string or int or float, e.g. `[2:5]`, 5, 5.0, default to None, it means run all examples
                         },
        )

        # Submit the task
        run_task(task_cfg=task_cfg)

        # Get the final report with summarizer
        report_list = Summarizer.get_report_from_cfg(task_cfg)
        logger.info(f'\n>>The report list:\n{report_list}')


if __name__ == '__main__':
    unittest.main()
