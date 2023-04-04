# Copyright (c) Alibaba, Inc. and its affiliates.

import os
from evals.tools import ItagManager


if __name__ == "__main__":

    tenant_id = '268ef75a'
    employee_id = '147543'
    token = """-----BEGIN RSA PRIVATE KEY-----
MIICXAIBAAKBgQCKlhCKIjdvh9486jOPg29GiFgmTvVQ5SxWteiR7gWAPphdVa9/
Iyzl2da6tRISST6ChksbEqHChSdcefSopATNP9jPsDVeuCTAmdkZktZDePEevbla
NQBYNjxKXFAY3v11PLJaofQ096IplOhhL7ECOL/94PC9Cox/scTc+HdsXwIDAQAB
AoGASfg288O3mwwGDrVit+MLbbYwdqIGRhtMQyvs6pcE0KKYaJjnhxCbUkOnXRhw
gNofR0OuqtCTDmRL0gw2Dh0dgjMR4wrrP+vyEamPHs0tj2+8FVrouWN+i/XxLBsy
uddkutpTwracbgU7Hdw8+Mu7zhs5PMR4XArjd3bUenq7lWECQQDkdONYlwPGGOTb
gDyyq9irrzp9Bjm8KFY8HGxtwhOuB+UYDvRcLOcQJ57gwNL9AT3ZPYVAlmP4oszl
laZ6OiadAkEAm0tn0ETJI2MzKHWMbco/fwqG2OEdKZmSDzj/iRmeqJyr0Fb1MxQs
QaJF6jPfSA1XmTaX6Fu/06ak4C3MUpOwKwJAEOZUqwkAznao91PVKaJstMaRnQ4I
11Jkjq3Ll5LzwbvzxoPUr7zimt9TcWzSLsUYvik+4jg9zPa+EX2wgvoqQQJALbnB
QpjOZMYTzSj3hWhU0/JkjD2UmagnUqYkz9ikV99x07GXF0gsU9MVJQXLC+spzOo/
RmKllAtwZrX1gKcN6wJBAKvLmLC/yQ0q50tPvnrJ4ZIB6sWqW7rB8O6IRPhL1nw1
yfMSYb/Yd8r/jyy8ANgaQ0kgpPNsOY6azET9R0Copv4=
-----END RSA PRIVATE KEY-----
    """

    itag_manager = ItagManager(tenant_id=tenant_id, token=token, employee_id=employee_id)

    # Create itag task from local csv file.
    template_id = '1642827551965851648'
    task_name = 'task_test_llm_evals_rank'
    dataset_name = 'your-dataset-name'
    dataset_path = os.path.join(os.path.dirname('__file__'), 'datasets/llm_evals_datasets_rank.csv')

    # itag_manager.process(dataset_path=dataset_path,
    #                      dataset_name=dataset_name,
    #                      template_id=template_id,
    #                      task_name=task_name)

    task_id='1642108430672875520'
    itag_manager.get_tag_task_result(task_id=task_id)
