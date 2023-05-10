# Copyright (c) Alibaba, Inc. and its affiliates.

import os

from evals.constants import DumpMode, TaskEnvs, DEFAULT_WORK_DIR
from evals.task import EvalTask
from evals.tools import ItagManager

""" This is an example of running a evaluation task pipeline. """


if __name__ == '__main__':

    # Step0: Set runtime envs
    cache_root_dir = os.environ.get(TaskEnvs.WORK_DIR, DEFAULT_WORK_DIR)

    # Step1: Get raw samples
    ...

    # Step2: Formatting samples
    ...

    # Step3: Generate prompts
    ...
    prompts_file = os.path.join(os.getcwd(), '..', 'evals/registry/data/poetry_gen/samples.jsonl')

    # Step4: Generate task config (yaml or dict)
    task_cfg_file = os.path.join(os.getcwd(), '..', 'evals/registry/tasks/task_qwen_gen_poetry.yaml')

    # Step5: run task
    eval_task = EvalTask(prompts=prompts_file, task_cfg=task_cfg_file)
    eval_task.run(num_processes=1,
                  chunksize=1,
                  dump_mode=DumpMode.OVERWRITE)
    print('Dump eval result to: ', eval_task.eval_results_path)

    # Step6 [Optional]: iTag pipeline (if you want to use iTag to tag your samples)
    IS_RUN_ITAG = False
    if IS_RUN_ITAG:

        #   Step6-1: Create iTag dataset and task
        itag_token = """-----BEGIN RSA PRIVATE KEY-----
            xxx
            -----END RSA PRIVATE KEY-----
            """
        itag_manager_args = dict(
            tenant_id='268ef75a',
            employee_id='147543',
            token=itag_token,
        )
        itag_manager = ItagManager(**itag_manager_args)

        itag_run_args = dict(
            template_id='1642827551965851648',
            task_name='task_test_llm_evals_rank_v1',
            dataset_name='dataset_test_llm_evals_rank_v1',
            dataset_path=os.path.join(os.getcwd(), '..', 'evals/tools/itag/datasets/llm_evals_datasets_rank.csv'),
        )
        itag_task_resp = itag_manager.process(**itag_run_args)

        #   Step6-2: Get iTag task results
        task_id = itag_task_resp.get('TaskId')
        df_res = itag_manager.get_tag_task_result(task_id=task_id)
        itag_result_file = os.path.join(cache_root_dir,
                                        'tasks/task_moss_gen_poetry_dev_v0',
                                        'task_test_llm_evals_rank_v1_out.csv')
        df_res.to_csv(itag_result_file, index=False)

    # Step7 [Optional]: Generate evaluation resport
    ...
