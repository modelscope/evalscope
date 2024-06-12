# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import Union

from llmuses.third_party.toolbench_static.infer import InferArgs, run_infer
from llmuses.third_party.toolbench_static.eval import EvalArgs, run_eval
from llmuses.utils.utils import yaml_to_dict


def run_task(task_config: Union[str, dict]):

    if isinstance(task_config, str):
        task_config: dict = yaml_to_dict(task_config)

    # Run inference for each domain
    infer_args: dict = task_config['infer_args']
    for domain, cfg_d in infer_args.items():
        print(f'** Infer domain: {domain}, cfg: {cfg_d}')
        task_infer_args = InferArgs(**cfg_d)
        run_infer(task_infer_args)

    # Run evaluation for each domain
    eval_args: dict = task_config['eval_args']
    for domain, cfg_d in eval_args.items():
        print(f'** Eval domain: {domain}, cfg: {cfg_d}')
        task_eval_args = EvalArgs(**cfg_d)
        run_eval(task_eval_args)


if __name__ == '__main__':
    task_cfg_file = 'config_default.yaml'

    run_task(task_config=task_cfg_file)
