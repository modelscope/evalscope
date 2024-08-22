# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import Union

from evalscope.third_party.longbench_write.infer import run_infer
from evalscope.third_party.longbench_write.eval import run_eval
from evalscope.utils import yaml_to_dict, json_to_dict
from evalscope.utils import get_logger

logger = get_logger()

"""
Entry file for LongWriter evaluation.
"""


def run_task(task_cfg: Union[str, dict]):

    # todo: logger task_cfg attributes

    if isinstance(task_cfg, str):
        if task_cfg.endswith('.yaml'):
            task_cfg: dict = yaml_to_dict(task_cfg)
        elif task_cfg.endswith('.json'):
            task_cfg: dict = json_to_dict(task_cfg)
        else:
            raise ValueError(f'Unsupported file format: {task_cfg}, should be yaml or json file.')

    # Parse task configuration





    # Run inference for specific model



    run_infer()


if __name__ == '__main__':
    # task_cfg = 'default_task.yaml'
    # task_cfg = 'default_task.json'

    # run_infer(model_id='ZhipuAI/LongWriter-glm4-9b',
    #           data_path='longbench_write.jsonl',
    #           output_dir='outputs',
    #           generation_kwargs=dict({
    #               'max_new_tokens': 32768,
    #               'temperature': 0.5})
    #           )

    # model_id: str,
    # pred_path: str,
    # output_dir: str,
    # prompt_template_path: str,
    # openai_api_key: str,
    # openai_gpt_model: str,
    # generation_kwargs: dict,
    # proc_num: int,

    task_cfg = dict(stage=['infer', 'eval_l', 'eval_q'],
                    model_id='ZhipuAI/LongWriter-glm4-9b',
                    data_path='resources/longbench_write.jsonl',
                    output_dir='./outputs',
                    prompt_template_path='prompt_templates.json',
                    openai_api_key='api_key',
                    openai_gpt_model='gpt_model',
                    generation_kwargs=dict({
                        'max_new_tokens': 32768,
                        'temperature': 0.5
                    }),
                    proc_num=8)

    run_task(task_cfg=task_cfg)
