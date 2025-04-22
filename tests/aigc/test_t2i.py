from dotenv import dotenv_values

env = dotenv_values('.env')

import os
import subprocess
import unittest

from evalscope.config import TaskConfig
from evalscope.constants import EvalType, JudgeStrategy, OutputType
from evalscope.run import run_task
from evalscope.utils import is_module_installed, test_level_list
from evalscope.utils.logger import get_logger

os.environ['LOG_LEVEL'] = 'DEBUG'

logger = get_logger()


class TestRun(unittest.TestCase):
    @unittest.skipUnless(0 in test_level_list(), 'skip test in current test level')
    def test_run_general(self):
        from evalscope.config import TaskConfig

        task_cfg = TaskConfig(
            datasets=[
                'general_t2i'
            ],
            dataset_args={
                'general_t2i': {
                    'metric_list': [
                        'PickScore',
                        'CLIPScore',
                        'HPSv2Score',
                        'BLIPv2Score',
                        'ImageRewardScore',
                        'VQAScore',
                        'FGA_BLIP2Score',
                    ],
                    'dataset_id': 'custom_eval/multimodal/t2i/example.jsonl',
                }
            }
        )

        run_task(task_cfg=task_cfg)
