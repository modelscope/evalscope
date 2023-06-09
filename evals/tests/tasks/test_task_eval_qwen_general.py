# Copyright (c) Alibaba, Inc. and its affiliates.

import os
import unittest

from evals.tasks.task_eval_qwen_code import TaskQwenCodeEval
from evals.tasks.task_eval_qwen_generation import TaskQwenGenerationEval
from evals.tasks.task_eval_qwen_math import TaskQwenMathEval
from evals.utils.utils import test_level_list

DEFAULT_TEST_LEVEL = 0


def condition(test_level=DEFAULT_TEST_LEVEL):
    return test_level in test_level_list()


class TestTaskEvalQwenGeneral(unittest.TestCase):

    def setUp(self) -> None:
        # Eval code skill
        task_qwen_code_cfg = os.path.join(
            os.getcwd(), '../..', 'registry/tasks/task_qwen_code.yaml')
        code_prompts = os.path.join(
            os.getcwd(), '../..',
            'registry/data/code/code_test_v2_model_result.jsonl')
        self.code_task = TaskQwenCodeEval(
            prompts=code_prompts, task_cfg=task_qwen_code_cfg)

        # Eval math skill
        task_qwen_math_cfg = os.path.join(
            os.getcwd(), '../..', 'registry/tasks/task_qwen_math.yaml')
        math_prompts = os.path.join(
            os.getcwd(), '../..',
            'registry/data/math/math_test_v2_model_result.jsonl')

        self.math_task = TaskQwenMathEval(
            prompts=math_prompts, task_cfg=task_qwen_math_cfg)

        task_qwen_generation_cfg = os.path.join(
            os.getcwd(), '../..', 'registry/tasks/task_qwen_generation.yaml')
        generation_prompts = os.path.join(
            os.getcwd(), '../..',
            'registry/data/common_generation/rouge_test_v7_model_result.jsonl')
        self.generation_task = TaskQwenGenerationEval(
            prompts=generation_prompts, task_cfg=task_qwen_generation_cfg)

    def tearDown(self) -> None:
        ...

    @unittest.skipUnless(
        condition(test_level=2), 'skip test in current test level')
    def test_code_task(self):
        self.code_task.run()

    @unittest.skipUnless(
        condition(test_level=2), 'skip test in current test level')
    def test_math_task(self):
        self.math_task.run()

    @unittest.skipUnless(
        condition(test_level=0), 'skip test in current test level')
    def test_common_generation_task(self):
        self.generation_task.run()


if __name__ == '__main__':
    unittest.main()
