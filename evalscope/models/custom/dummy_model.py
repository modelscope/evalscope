# Copyright (c) Alibaba, Inc. and its affiliates.
import os
import time
from typing import List

from evalscope.utils.logger import get_logger
from .custom_model import CustomModel

logger = get_logger()
"""
This script is used to rewrite the evaluation results without re-running the model predictions.
"""


class DummyCustomModel(CustomModel):

    def __init__(self, config: dict = {'model_id': 'dummy-model'}, **kwargs):
        super(DummyCustomModel, self).__init__(config=config, **kwargs)

    def predict(self, prompts: List[dict], **kwargs):
        # ONLY FOR DUMMY IMPLEMENTATION, DO NOT EDIT OR USE IN PRODUCTION.

        response = ''

        res_d: dict = {
            'choices': [{
                'index': 0,
                'message': {
                    'content': response,
                    'role': 'assistant'
                }
            }],
            'created': time.time(),
            'model': self.config.get('model_id'),  # should be model_id
            'object': 'chat.completion',
            'usage': {
                'completion_tokens': 0,
                'prompt_tokens': 0,
                'total_tokens': 0
            }
        }

        return [res_d for _ in prompts]


if __name__ == '__main__':
    from evalscope.run import run_task
    from evalscope.utils.io_utils import yaml_to_dict

    # step1: 如果outputs做了迁移，需要修改outputs/eval_xxx 中的configs/task_output_config.yaml中的路径配置
    # step2: 执行此脚本，默认使用use_cache=True，实现免推理对eval结果进行刷新

    swift_model = DummyCustomModel(config={'model_id': 'swift-model-dummy'})

    task_cfg_file = '/path/to/eval_your_model_results/configs/task_output_config.yaml'

    task_cfg_d = yaml_to_dict(task_cfg_file)
    task_cfg_d.update({'model': swift_model})

    eval_results: dict = run_task(task_cfg=task_cfg_d)
    print('** Evaluation results finished !\n')
