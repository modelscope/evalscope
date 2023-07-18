# Copyright (c) Alibaba, Inc. and its affiliates.

from llmuses.task import EvalTask


class TaskQwenMathEval(EvalTask):

    def __init__(self, prompts, task_cfg):
        super().__init__(prompts=prompts, task_cfg=task_cfg)
