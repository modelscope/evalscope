# Copyright (c) Alibaba, Inc. and its affiliates.
from mmengine.config import read_base


with read_base():
    from llmuses.backend.opencompass.tasks.eval_datasets import datasets

# OpenAI API format evaluation needs a special humaneval postprocessor
from opencompass.datasets.humaneval import humaneval_gpt_postprocess
for _dataset in datasets:
    if _dataset['path'] == 'openai_humaneval':
        _dataset['eval_cfg']['pred_postprocessor']['type'] = humaneval_gpt_postprocess


# todo: oc 的main 传入run_task

if __name__ == '__main__':
    print('===start===')
    from modelscope import MsDataset

    print(f'>>datasets: {datasets[0]}')
