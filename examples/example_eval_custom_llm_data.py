# Copyright (c) Alibaba, Inc. and its affiliates.
import torch
from evalscope.run import run_task
from evalscope.config import TaskConfig, registry_tasks

def run_swift_eval():
    # Prepare the config
    data_pattern = 'general_qa'

    # 2. 配置自定义数据集文件
    TaskConfig.registry(
        name='custom_dataset',      # 任务名称
        data_pattern=data_pattern,  # 数据格式
        dataset_dir='custom_qa',       # 数据集路径
        subset_list=['example']     # 评测数据集名称，上述 example.jsonl
    )

    # 3. 配置任务，通过任务名称获取配置
    task_cfg = registry_tasks['custom_dataset']

    # 4. 配置模型和其他配置
    task_cfg.update({
        'model_args': {'revision': None, 'precision': torch.float16, 'device_map': 'auto'},
        'eval_type': 'checkpoint',                 # 评测类型，需保留，固定为checkpoint
        'model': '../models/Qwen2-0.5B-Instruct', # 模型路径
        'template_type': 'qwen',                  # 模型模板类型
        'outputs': 'outputs',
        'mem_cache': False,
        'limit': 10,
        })

    # Run task
    run_task(task_cfg=task_cfg)


if __name__ == '__main__':    
    run_swift_eval()
