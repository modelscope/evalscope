# Copyright (c) Alibaba, Inc. and its affiliates.

from llmuses.third_party.toolbench_static.infer import InferArgs, run_infer
from llmuses.third_party.toolbench_static.eval import EvalArgs, run_eval
from llmuses.utils.utils import yaml_to_dict


def main(config_file: str):
    task_config_d: dict = yaml_to_dict(config_file)

    infer_args: dict = task_config_d['infer_args']
    for domain, cfg_d in infer_args.items():
        print(f'** Infer domain: {domain}, cfg: {cfg_d}')
        task_infer_args = InferArgs(**cfg_d)
        run_infer(task_infer_args)

    eval_args: dict = task_config_d['eval_args']


if __name__ == '__main__':
    task_cfg_file = 'config_default.yaml'
    main(config_file=task_cfg_file)
