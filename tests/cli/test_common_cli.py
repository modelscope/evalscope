# Copyright (c) Alibaba, Inc. and its affiliates.
import subprocess
import tempfile
import unittest
import shutil

from llmuses.utils.test_utils import test_level
from llmuses.utils.logger import get_logger

logger = get_logger()


class TestCommonCli(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.test_root_dir = tempfile.mkdtemp()

        cls.model_id: str = 'ZhipuAI/chatglm3-6b'
        cls.template_type: str = 'chatglm3'
        cls.dataset_list: str = 'arc ceval'

    def tearDownClass(cls) -> None:
        shutil.rmtree(cls.test_root_dir)
        logger.info(f'==> Temporary directory {cls.test_root_dir} successfully removed!')

    @unittest.skipUnless(test_level() >= 1, '>> Skip test in current test level')
    def test_dry_run(self):
        logger.info('==> Test dry run ...')

        cmd = f'python3 -m llmuses.run ' \
              f'--model {self.model_id} ' \
              f'--template-type {self.template_type} ' \
              f'--datasets {self.dataset_list} ' \
              f'--dry-run'

        stat, output = subprocess.getstatusoutput(cmd)
        if stat != 0:
            logger.error(output)
        self.assertEqual(stat, 0)


if __name__ == '__main__':
    unittest.main()

