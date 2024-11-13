# Copyright (c) Alibaba, Inc. and its affiliates.

import subprocess
import unittest
from evalscope.utils import test_level_list, is_module_installed
from evalscope.utils.logger import get_logger
from evalscope.run import run_task

logger = get_logger()


class TestMTEB(unittest.TestCase):

    def setUp(self) -> None:
        self._check_env('mteb')

    def tearDown(self) -> None:
        pass

    @staticmethod
    def _check_env(module_name: str):
        if is_module_installed(module_name):
            logger.info(f'{module_name} is installed.')
        else:
            raise ModuleNotFoundError(f'run: pip install {module_name}')

    @unittest.skipUnless(0 in test_level_list(), 'skip test in current test level')
    def test_run_one_stage_mteb(self):
        task_cfg = {
            'eval_backend': 'RAGEval',
            'eval_config': {
                'tool': 'MTEB',
                'model': [
                    {
                        'model_name_or_path': 'AI-ModelScope/m3e-base',
                        'pooling_mode': None,  # load from model config
                        'max_seq_length': 512,
                        'prompt': '',
                        'model_kwargs': {'torch_dtype': 'auto'},
                        'encode_kwargs': {
                            'batch_size': 128,
                        },
                    }
                ],
                'eval': {
                    'tasks': [
                        'TNews',
                        'CLSClusteringS2S',
                        'T2Reranking',
                        'T2Retrieval',
                        'ATEC',
                    ],
                    'verbosity': 2,
                    'output_folder': 'outputs',
                    'overwrite_results': True,
                    'limits': 500,
                },
            },
        }

        run_task(task_cfg)

    @unittest.skipUnless(0 in test_level_list(), 'skip test in current test level')
    def test_run_two_stage_mteb(self):
        task_cfg = {
            'eval_backend': 'RAGEval',
            'eval_config': {
                'tool': 'MTEB',
                'model': [
                    {
                        'model_name_or_path': 'AI-ModelScope/m3e-base',
                        'is_cross_encoder': False,
                        'max_seq_length': 512,
                        'prompt': '',
                        'model_kwargs': {'torch_dtype': 'auto'},
                        'encode_kwargs': {
                            'batch_size': 64,
                        },
                    },
                    {
                        'model_name_or_path': 'OpenBMB/MiniCPM-Reranker',
                        'is_cross_encoder': True,
                        'max_seq_length': 512,
                        'prompt': '为这个问题生成一个检索用的表示',
                        'model_kwargs': {'torch_dtype': 'auto'},
                        'encode_kwargs': {
                            'batch_size': 32,
                        },
                    },
                ],
                'eval': {
                    'tasks': ['MedicalRetrieval', 'T2Retrieval'],
                    'verbosity': 2,
                    'output_folder': 'outputs',
                    'overwrite_results': True,
                    'limits': 10,
                },
            },
        }

        run_task(task_cfg)

    @unittest.skipUnless(0 in test_level_list(), 'skip test in current test level')
    def test_run_custom(self):
        task_cfg = {
            'eval_backend': 'RAGEval',
            'eval_config': {
                'tool': 'MTEB',
                'model': [
                    {
                        'model_name_or_path': 'AI-ModelScope/m3e-base',
                        'pooling_mode': None,  # load from model config
                        'max_seq_length': 512,
                        'prompt': '',
                        'model_kwargs': {'torch_dtype': 'auto'},
                        'encode_kwargs': {
                            'batch_size': 128,
                        },
                    }
                ],
                'eval': {
                    'tasks': ['CustomRetrieval'],
                    'dataset_path': 'custom_eval/text/retrieval',
                    'verbosity': 2,
                    'output_folder': 'outputs',
                    'overwrite_results': True,
                    'limits': 500,
                },
            },
        }

        run_task(task_cfg)


if __name__ == '__main__':
    unittest.main(buffer=False)
