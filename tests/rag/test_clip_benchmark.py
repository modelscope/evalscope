# Copyright (c) Alibaba, Inc. and its affiliates.
import os
import subprocess
import unittest
from pathlib import Path

import pytest

pytestmark = pytest.mark.timeout(600)

from evalscope.backend.rag_eval.clip_benchmark.utils import webdataset_convert
from evalscope.run import run_task
from evalscope.utils.import_utils import is_module_installed
from evalscope.utils.logger import get_logger
from tests.utils import test_level_list

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
                    'dataset_name': [
                        'muge',
                        'mnist',
                        'flickr8k'
                    ],
                    'split': 'test',
                    'batch_size': 128,
                    'num_workers': 1,
                    'verbose': True,
                    'skip_existing': False,
                    'cache_dir': 'cache',
                    'limit': 10,
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
                    'limit': 10,
                },
            },
        }

        run_task(task_cfg)


def test_webdataset_converter_writes_metadata_files(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    class DummyShardWriter:

        shard = 1

        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        def write(self, sample: object) -> None:
            pass

        def close(self) -> None:
            pass

    monkeypatch.setattr(webdataset_convert.torch.utils.data, 'DataLoader', lambda dataset, **_: dataset)
    monkeypatch.setattr(webdataset_convert.webdataset, 'ShardWriter', DummyShardWriter)
    monkeypatch.setattr(webdataset_convert, 'tqdm', lambda iterator, **_: iterator)

    dataset = type('Dataset', (list,), {'classes': ['cat', 'dog'], 'templates': ['a photo of a {c}']})([(b'input', 1)])
    webdataset_convert.convert_dataset(dataset, 'train', str(tmp_path), image_format='bin', multilabel=True)

    assert (tmp_path / 'classnames.txt').read_text() == 'cat\ndog\n'
    assert (tmp_path / 'zeroshot_classification_templates.txt').read_text() == 'a photo of a {c}\n'
    assert (tmp_path / 'dataset_type.txt').read_text() == 'multilabel\n'
    assert (tmp_path / 'train' / 'nshards.txt').read_text() == '1\n'

    webdataset_convert.convert_retrieval_dataset([(b'input', ['caption'])], 'validation', str(tmp_path), image_format='bin')

    assert (tmp_path / 'dataset_type.txt').read_text() == 'retrieval\n'
    assert (tmp_path / 'validation' / 'nshards.txt').read_text() == '1\n'


if __name__ == '__main__':
    unittest.main(buffer=False)
