# Copyright (c) Alibaba, Inc. and its affiliates.
import torch
from evalscope.run import run_task
from evalscope.config import TaskConfig, registry_tasks
from evalscope.summarizer import Summarizer

def run_swift_eval():
    # Prepare the config

    data_pattern = 'ceval' # 
    
    TaskConfig.registry(
        name='custom_dataset',
        data_pattern=data_pattern,
        dataset_dir='custom',
        subset_list=['example']
    )
    
    task_cfg = registry_tasks['custom_dataset']
    
    task_cfg.update({
        'model_args': {'revision': None, 'precision': torch.float16, 'device_map': 'auto'},
        'model': '../models/Qwen2-0.5B-Instruct',
        'template_type': 'qwen', 
        'outputs': 'outputs',
        'mem_cache': False,
        'limit': 10,
        'eval_type': 'checkpoint'
    })

    # Run task
    run_task(task_cfg=task_cfg)

    print('>> Start to get the report with summarizer ...')
    report_list = Summarizer.get_report_from_cfg(task_cfg)
    print(f'\n>>The report list: {report_list}')


if __name__ == '__main__':    run_swift_eval()
