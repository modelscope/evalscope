# Copyright (c) Alibaba, Inc. and its affiliates.
import os
import time
from typing import List

import json

from evalscope.constants import DEFAULT_WORK_DIR
from evalscope.models.custom import CustomModel
from evalscope.run import run_task
from evalscope.summarizer import Summarizer
from evalscope.utils import yaml_to_dict
from evalscope.utils.logger import get_logger

logger = get_logger()


class SwiftModel(CustomModel):

    def __init__(self, config: dict, **kwargs):

        super(SwiftModel, self).__init__(config=config, **kwargs)

    def predict(self, prompts: str, **kwargs):

        # query = '浙江的省会在哪里？'
        # prompts = [query]
        # response, history = self.inference(self.model, self.template, prompts)
        # response: str = str(response)

        # ONLY FOR TEST
        response = 'The answer is C.'

        res_d: dict = {
            'choices': [{
                'index': 0,
                'message': {
                    # 'content': f'The answer is B. Raw prompt: {prompt}',
                    'content': response,
                    'role': 'assistant'
                }
            }],
            'created':
            time.time(),
            'model':
            self.config.get('model_id'),  # should be model_id
            'object':
            'chat.completion',
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
            'work_dir': DEFAULT_WORK_DIR,
            'outputs': './outputs/eval_swift_dummy',
            'mem_cache': False,
            'dataset_hub': 'ModelScope',
            'dataset_dir': DEFAULT_WORK_DIR,
            'stage': 'all',
            'limit': 10,
            'debug': False
        }

    return task_cfg_d


if __name__ == '__main__':

    from evalscope.config import TaskConfig

    swift_model = SwiftModel(config={'model_id': 'swift_grok-base-dummy'})
    print(TaskConfig.list())  # ['arc', 'gsm8k']   # 'arc', 'gsm8k', 'bbh_mini', 'mmlu_mini', 'ceval_mini'

    # Customize your own dataset, refer to datasets:
    # wget https://modelscope.oss-cn-beijing.aliyuncs.com/open_data/benchmark/data.zip
    # unzip data.zip
    custom_dataset_name = 'general_qa_swift_custom_dataset'
    custom_dataset_pattern = 'general_qa'  # 可选范围： ['arc', 'gsm8k', 'mmlu', 'ceval', 'bbh']
    TaskConfig.registry(
        name=custom_dataset_name,
        data_pattern=custom_dataset_pattern,
        dataset_dir='/path/to/general_qa_swift',
        # subset_list=['my_swift_custom_subset1', 'my_swift_custom_subset2'],
    )

    # Load the task config list
    task_config_list = TaskConfig.load(custom_model=swift_model, tasks=[custom_dataset_name, 'arc'])

    # You can update the task_config with your own settings
    for config_item in task_config_list:
        config_item.limit = 20  # Note: limit the number of each subset to evaluate; default is None
        config_item.use_cache = False

    print(task_config_list)

    eval_results: dict = run_task(task_cfg=task_config_list)
    print('** Evaluation results finished !\n')

    # Get the final report for your evaluation task
    final_report: List[dict] = Summarizer.get_report_from_cfg(task_cfg=task_config_list)
    print(f'*** Final report ***\n {json.dumps(final_report, ensure_ascii=False)}\n')
