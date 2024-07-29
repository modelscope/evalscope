# Copyright (c) Alibaba, Inc. and its affiliates.
import os
import time

from evalscope.models.custom import CustomModel
from evalscope.run import run_task
from evalscope.constants import DEFAULT_ROOT_CACHE_DIR
from evalscope.utils import yaml_to_dict
from evalscope.utils.logger import get_logger

logger = get_logger()

"""
This script is used to rewrite the evaluation results without re-running the model predictions.
"""


class DummyCustomModel(CustomModel):

    def __init__(self, config: dict, **kwargs):
        super(DummyCustomModel, self).__init__(config=config, **kwargs)

    def predict(self, prompts: str, **kwargs):
        # ONLY FOR DUMMY IMPLEMENTATION, DO NOT EDIT OR USE IN PRODUCTION.

        response = 'The answer is C. NOTE: ONLY FOR TEST'

        res_d: dict = {
            'choices': [
                {
                    'index': 0,
                    'message': {
                        # 'content': f'The answer is B. Raw prompt: {prompt}',
                        'content': response,
                        'role': 'assistant'
                    }
                }
            ],
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


def get_task_cfg(cfg_file: str, model_instance: CustomModel):
    if cfg_file:
        cfg_file: str = os.path.abspath(cfg_file)
        logger.info(f'Loading task config from {cfg_file}')
        task_cfg_d: dict = yaml_to_dict(yaml_file=cfg_file)
        task_cfg_d.update({'model': model_instance})
        logger.info(f'**Task config: {task_cfg_d}')
    else:
        # 默认config 示例
        task_cfg_d = {
            'model_args': {},
            'generation_config': {},
            'dataset_args': {},
            'dry_run': False,
            'model': model_instance,  # NOTE: model_id or # model_dir or model_instance(CustomModel)
            'eval_type': 'custom',  # NOTE: `checkpoint` or `custom` or `service`
            'datasets': ['arc'],
            'work_dir': DEFAULT_ROOT_CACHE_DIR,
            'outputs': './outputs/eval_swift_dummy',
            'mem_cache': False,
            'dataset_hub': 'ModelScope',
            'dataset_dir': DEFAULT_ROOT_CACHE_DIR,
            'stage': 'all',
            'limit': 10,
            'debug': False
        }

    return task_cfg_d


if __name__ == '__main__':
    # step1: 如果outputs做了迁移，需要修改outputs/eval_xxx 中的configs/task_output_config.yaml中的路径配置
    # step2: 执行此脚本，默认使用use_cache=True，实现免推理对eval结果进行刷新

    swift_model = DummyCustomModel(config={'model_id': 'swift-model-dummy'})

    task_cfg_file = '/path/to/eval_your_model_results/configs/task_output_config.yaml'

    task_cfg_d = yaml_to_dict(task_cfg_file)
    task_cfg_d.update({'model': swift_model})

    eval_results: dict = run_task(task_cfg=task_cfg_d)
    print(f'** Evaluation results finished !\n')

