# Copyright (c) Alibaba, Inc. and its affiliates.

import subprocess
import unittest
from evalscope.utils import test_level_list, is_module_installed
from evalscope.utils.logger import get_logger

logger = get_logger()


class TestRun(unittest.TestCase):

    def setUp(self) -> None:
        logger.info(f'Init env for evalscope native run UTs ...\n')
        self._check_env('evalscope')

    def tearDown(self) -> None:
        pass

    @staticmethod
    def _check_env(module_name: str):
        if is_module_installed(module_name):
            logger.info(f'{module_name} is installed.')
        else:
            raise ModuleNotFoundError(f'run: pip install {module_name}')

    @unittest.skipUnless(0 in test_level_list(), 'skip test in current test level')
    def test_run_simple_eval(self):
        model = 'ZhipuAI/chatglm3-6b'
        template_type = 'chatglm3'
        datasets = 'arc'  # arc ceval
        limit = 100

        cmd_simple = f'python3 -m evalscope.run ' \
                     f'--model {model} ' \
                     f'--template-type {template_type} ' \
                     f'--datasets {datasets} ' \
                     f'--limit {limit}'

        logger.info(f'Start to run command: {cmd_simple}')
        run_res = subprocess.run(cmd_simple, text=True, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        assert run_res.returncode == 0, f'Failed to run command: {cmd_simple}'
        logger.info(f'>>test_run_simple_eval stdout: {run_res.stdout}')
        logger.error(f'>>test_run_simple_eval stderr: {run_res.stderr}')

    @unittest.skipUnless(0 in test_level_list(), 'skip test in current test level')
    def test_run_eval_with_args(self):
        model = 'ZhipuAI/chatglm3-6b'
        template_type = 'chatglm3'
        datasets = 'arc ceval'  # arc ceval
        limit = 5
        dataset_args = '{"ceval": {"few_shot_num": 0, "few_shot_random": false}}'

        cmd_with_args = f'python3 -m evalscope.run ' \
                        f'--model {model} ' \
                        f'--template-type {template_type} ' \
                        f'--datasets {datasets} ' \
                        f'--limit {limit} ' \
                        f'--generation-config do_sample=false,temperature=0.0 ' \
                        f"""--dataset-args \'{dataset_args}\' """

        logger.info(f'Start to run command: {cmd_with_args}')
        run_res = subprocess.run(cmd_with_args, text=True, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        assert run_res.returncode == 0, f'Failed to run command: {cmd_with_args}'
        logger.info(f'>>test_run_eval_with_args stdout: {run_res.stdout}')
        logger.error(f'>>test_run_eval_with_args stderr: {run_res.stderr}')

    @unittest.skipUnless(0 in test_level_list(), 'skip test in current test level')
    def test_run_eval_local(self):
        ...


if __name__ == '__main__':
    unittest.main()
