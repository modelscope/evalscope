# Copyright (c) Alibaba, Inc. and its affiliates.

import os

from evals.tools import ItagManager

if __name__ == '__main__':

    tenant_id, employee_id = open(
        'private/itag_info.txt').readlines().strip().split(',')
    token = """-----BEGIN RSA PRIVATE KEY-----
xxx
xxx
-----END RSA PRIVATE KEY-----
    """

    itag_manager = ItagManager(
        tenant_id=tenant_id, token=token, employee_id=employee_id)

    # Create itag task from local csv file.
    # template_id = '1642827551965851648'
    # task_name = 'task_test_llm_evals_rank'
    # dataset_name = 'your-dataset-name'
    # dataset_path = os.path.join(os.path.dirname('__file__'), 'datasets/llm_evals_datasets_rank.csv')
    #
    # itag_manager.process(dataset_path=dataset_path,
    #                      dataset_name=dataset_name,
    #                      template_id=template_id,
    #                      task_name=task_name)

    # Get tag result
    task_id = '1642108430672875520'
    df_res = itag_manager.get_tag_task_result(task_id=task_id)
    df_res.to_csv('llm_evals_rank.csv', index=False)
