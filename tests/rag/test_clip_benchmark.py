# Copyright (c) Alibaba, Inc. and its affiliates.

import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import subprocess
import unittest

from evalscope.run import run_task
from evalscope.utils import is_module_installed, test_level_list
from evalscope.utils.logger import get_logger

logger = get_logger()


class TestCLIPBenchmark(unittest.TestCase):

    def setUp(self) -> None:
        self._check_env('webdataset')

    def tearDown(self) -> None:
        pass

    @staticmethod
    def _check_env(module_name: str):
        if is_module_installed(module_name):
            logger.info(f'{module_name} is installed.')
        else:
            raise ModuleNotFoundError(f'run: pip install {module_name}')

    @unittest.skipUnless(0 in test_level_list(), 'skip test in current test level')
    def test_run_task(self):
        task_cfg = {
            'eval_backend': 'RAGEval',
            'eval_config': {
                'tool': 'clip_benchmark',
                'eval': {
                    'models': [
                        {
                            'model_name': 'AI-ModelScope/chinese-clip-vit-large-patch14-336px',
                        }
                    ],
                    'dataset_name': ['muge', 'mnist'],
                    'split': 'test',
                    'batch_size': 128,
                    'num_workers': 1,
                    'verbose': True,
                    'skip_existing': False,
                    'cache_dir': 'cache',
                    'limit': 1000,
                },
            },
        }

        run_task(task_cfg)

    @unittest.skipUnless(0 in test_level_list(), 'skip test in current test level')
    def test_run_custom(self):
        task_cfg = {
            'eval_backend': 'RAGEval',
            'eval_config': {
                'tool': 'clip_benchmark',
                'eval': {
                    'models': [
                        {
                            'model_name': 'AI-ModelScope/chinese-clip-vit-large-patch14-336px',
                        }
                    ],
                    'dataset_name': ['custom'],
                    'data_dir': 'custom_eval/multimodal/text-image-retrieval',
                    'split': 'test',
                    'batch_size': 128,
                    'num_workers': 1,
                    'verbose': True,
                    'skip_existing': False,
                    'limit': 1000,
                },
            },
        }

        run_task(task_cfg)


if __name__ == '__main__':
    unittest.main(buffer=False)
