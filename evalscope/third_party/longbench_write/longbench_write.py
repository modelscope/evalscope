# Copyright (c) Alibaba, Inc. and its affiliates.
import os
from typing import Union

from evalscope.third_party.longbench_write.eval import run_eval
from evalscope.third_party.longbench_write.infer import run_infer
from evalscope.utils import get_logger
from evalscope.utils.io_utils import json_to_dict, yaml_to_dict

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
    model: str = task_cfg.get('model')
    input_data_path: str = task_cfg.get('input_data_path')
    output_dir: str = task_cfg.get('output_dir')

    infer_config: dict = task_cfg.get('infer_config')
    eval_config: dict = task_cfg.get('eval_config')
    assert infer_config is not None and eval_config is not None, 'Please provide infer_config and eval_config.'

    # Run inference process
    pred_res_path = run_infer(model=model,
                              data_path=input_data_path or os.path.join(os.path.dirname(__file__), 'resources/longbench_write.jsonl'),
                              output_dir=output_dir,
                              api_config=dict(
                                  openai_api_key=infer_config.get('openai_api_key'),
                                  openai_api_base=infer_config.get('openai_api_base'),
                                  is_chat=infer_config.get('is_chat', True),
                                  verbose=infer_config.get('verbose', False),
                              ),
                              generation_kwargs=infer_config.get('generation_kwargs'),
                              enable='infer' in stage,
                              proc_num=infer_config.get('proc_num', 16))

    # Run eval process
    run_eval(model=model,
             pred_path=pred_res_path,
             output_dir=output_dir,
             prompt_template_path=os.path.join(os.path.dirname(__file__), 'resources/judge.txt'),
             openai_api_key=eval_config.get('openai_api_key'),
             openai_api_base=eval_config.get('openai_api_base'),
             openai_gpt_model=eval_config.get('openai_gpt_model'),
             generation_kwargs=eval_config.get('generation_kwargs'),
             proc_num=eval_config.get('proc_num', 16),
             stage=stage)


if __name__ == '__main__':
    # Note: evaluation task configuration can also be loaded from yaml or json file.
    # task_cfg = os.path.join(os.path.dirname(__file__), 'default_task.yaml')
    # task_cfg = os.path.join(os.path.dirname(__file__), 'default_task.json')
    task_cfg = dict(stage=['infer', 'eval_l', 'eval_q'],
                    model='ZhipuAI/LongWriter-glm4-9b',  # or /path/to/your_model_dir
                    input_data_path=None,
                    output_dir='./outputs',

                    infer_config={
                        'openai_api_base': 'http://127.0.0.1:8000/v1/chat/completions',
                        'is_chat': True,
                        'verbose': False,
                        'generation_kwargs': {'max_new_tokens': 32768, 'temperature': 0.5, 'repetition_penalty': 1.0},
                    },

                    eval_config={
                        'openai_api_key': None,
                        'openai_api_base': 'https://api.openai.com/v1/chat/completions',
                        'openai_gpt_model': 'gpt-4o-2024-05-13',
                        'generation_kwargs': {'max_new_tokens': 1024, 'temperature': 0.5, 'stop': None},
                        'proc_num': 16,
                    },
                    )

    run_task(task_cfg=task_cfg)
