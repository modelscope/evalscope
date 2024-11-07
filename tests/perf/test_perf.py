# Copyright (c) Alibaba, Inc. and its affiliates.

import unittest

from evalscope.perf.main import run_perf_benchmark
from evalscope.utils import test_level_list


class TestPerf(unittest.TestCase):

    def setUp(self) -> None:
        pass

    def tearDown(self) -> None:
        pass

    @unittest.skipUnless(0 in test_level_list(), 'skip test in current test level')
    def test_run_perf(self):
        task_cfg = {
            'url': 'http://127.0.0.1:8000/v1/chat/completions',
            'parallel': 1,
            'model': 'qwen2.5',
            'number': 15,
            'api': 'openai',
            'dataset': 'openqa',
            'stream': True
        }
        run_perf_benchmark(task_cfg)


if __name__ == '__main__':
    unittest.main(buffer=False)
