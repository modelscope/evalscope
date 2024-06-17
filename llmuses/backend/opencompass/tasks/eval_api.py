# Copyright (c) Alibaba, Inc. and its affiliates.
from mmengine.config import read_base
from opencompass.datasets.humaneval import humaneval_gpt_postprocess
from opencompass.cli.arguments import ModelConfig


with read_base():
    from llmuses.backend.opencompass.tasks.eval_datasets import datasets

# 1. Get datasets
# Note: OpenAI API format evaluation needs a special humaneval postprocessor
for _dataset in datasets:
    if _dataset['path'] == 'openai_humaneval':
        _dataset['eval_cfg']['pred_postprocessor']['type'] = humaneval_gpt_postprocess

    # TODO: consider the custom dataset name
    _dataset['dataset_name'] = _dataset['type'].__module__.split('.')[-1]


# 2. Get api meta_template
# TODO: Chat model and base model ?
api_meta_template = dict(
    round=[
        dict(role='HUMAN', api_role='HUMAN'),
        dict(role='BOT', api_role='BOT', generate=True)
    ],
    reserved_roles=[
        dict(role='SYSTEM', api_role='SYSTEM'),
    ],
)

# 3. Get models
# model_config = ModelConfig(abbr='', path='', meta_template='', openai_api_base='')
# models.append(asdict(model_config))
models = []   # TODO: to be passed

# todo: oc 的main 传入run_task

if __name__ == '__main__':
    print('===start===')

    print(datasets)
