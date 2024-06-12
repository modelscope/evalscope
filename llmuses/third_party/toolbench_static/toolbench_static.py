# Copyright (c) Alibaba, Inc. and its affiliates.
import os
from typing import Union
from copy import deepcopy

from llmuses.third_party.toolbench_static.infer import InferArgs, run_infer
from llmuses.third_party.toolbench_static.eval import EvalArgs, run_eval
from llmuses.utils.utils import yaml_to_dict


def run_task(task_config: Union[str, dict]):

    if isinstance(task_config, str):
        task_config: dict = yaml_to_dict(task_config)

    # Run inference for each domain
    infer_args: dict = task_config['infer_args']
    for domain in ['in_domain', 'out_of_domain']:
        domain_infer_args = deepcopy(infer_args)
        domain_infer_args.update({'data_path': os.path.join(infer_args['data_path'], f'{domain}.json')})
        domain_infer_args.update({'output_dir': os.path.join(infer_args['output_dir'], domain)})

        task_infer_args = InferArgs(**domain_infer_args)
        print(f'**Run infer config: {task_infer_args}')
        run_infer(task_infer_args)

    # Run evaluation for each domain
    eval_args: dict = task_config['eval_args']
    for domain in ['in_domain', 'out_of_domain']:
        domain_eval_args = deepcopy(eval_args)
        domain_eval_args.update({'input_path': os.path.join(eval_args['input_path'], domain)})
        domain_eval_args.update({'output_path': os.path.join(eval_args['output_path'], domain)})

        task_eval_args = EvalArgs(**domain_eval_args)
        print(f'**Run eval config: {task_eval_args}')
        run_eval(task_eval_args)


if __name__ == '__main__':
    task_cfg_file = 'config_default.yaml'

    run_task(task_config=task_cfg_file)
