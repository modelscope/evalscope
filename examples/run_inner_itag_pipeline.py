# Copyright (c) Alibaba, Inc. and its affiliates.

import os
from evals.tools import ItagManager


if __name__ == "__main__":

    tenant_id = 'xxx'
    employee_id = 'xxx'

    token = """-----BEGIN RSA PRIVATE KEY-----
xxx
xxx
-----END RSA PRIVATE KEY-----
    """

    itag_manager = ItagManager(tenant_id=tenant_id, token=token, employee_id=employee_id)

    # Create dataset from local file
    dataset_path = os.path.join(os.path.dirname('__file__'), 'datasets/llm_evals_datasets_rank.csv')
    itag_manager.create_dataset(dataset_path)

    # Get dataset info
    dataset_info = itag_manager.get_dataset_info(dataset_id=329240)
    print('>>dataset_info: ', dataset_info)

    # Create itag task
    template_id = 'xxx'
    task_name = 'xxx'
    dataset_id = 'xxx'

    create_task_response = itag_manager.create_tag_task(task_name=task_name,
                                                        dataset_id=dataset_id,
                                                        template_id=template_id)
    print('>>create_task_response: ', create_task_response)

    # Get task result info
    task_id = 'xxx'
    task_info = itag_manager.get_tag_task_info(task_id=task_id)
    print('>>task_info: ', task_info)
