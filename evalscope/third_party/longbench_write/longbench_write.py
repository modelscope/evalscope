# Copyright (c) Alibaba, Inc. and its affiliates.
import os
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

    if isinstance(task_cfg, str):
        if task_cfg.endswith('.yaml'):
            task_cfg: dict = yaml_to_dict(task_cfg)
        elif task_cfg.endswith('.json'):
            task_cfg: dict = json_to_dict(task_cfg)
        else:
            raise ValueError(f'Unsupported file format: {task_cfg}, should be yaml or json file.')

    # Parse task configuration
    stage: list = task_cfg.get('stage', ['infer', 'eval_l', 'eval_q'])
    model_id_or_path: str = task_cfg.get('model_id_or_path')
    output_dir: str = task_cfg.get('output_dir')
    openai_api_key: str = os.getenv('OPENAI_API_KEY') or task_cfg.get('openai_api_key')
    openai_gpt_model: str = task_cfg.get('openai_gpt_model')
    infer_generation_kwargs = task_cfg.get('infer_generation_kwargs')
    eval_generation_kwargs = task_cfg.get('eval_generation_kwargs')
    proc_num: int = task_cfg.get('proc_num', 8)

    # Run inference process
    pred_res_path = run_infer(model_id_or_path=model_id_or_path,
                              data_path=os.path.join(os.path.dirname(__file__), 'resources/longbench_write.jsonl'),
                              output_dir=output_dir,
                              generation_kwargs=infer_generation_kwargs,
                              enable='infer' in stage)

    # Run eval process
    run_eval(model_id_or_path=model_id_or_path,
             pred_path=pred_res_path,
             output_dir=output_dir,
             prompt_template_path=os.path.join(os.path.dirname(__file__), 'resources/judge.txt'),
             openai_api_key=openai_api_key,
             openai_gpt_model=openai_gpt_model,
             generation_kwargs=eval_generation_kwargs,
             proc_num=proc_num,
             stage=stage)


if __name__ == '__main__':
    # task_cfg = os.path.join(os.path.dirname(__file__), 'default_task.yaml')
    # task_cfg = os.path.join(os.path.dirname(__file__), 'default_task.json')

    task_cfg = dict(stage=['infer', 'eval_l', 'eval_q'],
                    model_id_or_path='ZhipuAI/LongWriter-glm4-9b',  # or /path/to/your_model_dir
                    output_dir='./outputs',
                    openai_api_key=None,
                    openai_gpt_model='gpt-4o-2024-05-13',
                    infer_generation_kwargs={
                        'max_new_tokens': 32768,
                        'temperature': 0.5
                    },
                    eval_generation_kwargs={
                        'max_new_tokens': 1024,
                        'temperature': 0.5,
                        'stop': None
                    },
                    proc_num=8)

    run_task(task_cfg=task_cfg)
