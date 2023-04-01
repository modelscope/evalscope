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

    # Create itag task from local csv file.
    template_id = 'xxx'
    task_name = 'your-itag-task-name'
    dataset_path = os.path.join(os.path.dirname('__file__'), 'datasets/llm_evals_datasets_rank.csv')

    itag_manager.process(dataset_path=dataset_path, template_id=template_id, task_name=task_name)
