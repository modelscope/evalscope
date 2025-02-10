# Copyright (c) Alibaba, Inc. and its affiliates.
from evalscope.config import TaskConfig
from evalscope.run import run_task


def run_swift_eval():

    task_cfg = TaskConfig(
        model='qwen/Qwen2-0.5B-Instruct',
        datasets=['general_mcq', 'general_qa'],  # 数据格式，选择题格式固定为 'ceval'
        dataset_args={
            'general_mcq': {
                'local_path': 'custom_eval/text/mcq',  # 自定义数据集路径
                'subset_list': [
                    'example'  # 评测数据集名称
                ]
            },
            'general_qa': {
                'local_path': 'custom_eval/text/qa',  # 自定义数据集路径
                'subset_list': [
                    'example'  # 评测数据集名称
                ]
            }
        },
    )
    run_task(task_cfg=task_cfg)


if __name__ == '__main__':
    run_swift_eval()
