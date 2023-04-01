# Copyright (c) Alibaba, Inc. and its affiliates.
import uuid

from alibabacloud_tea_openapi import models as open_api_models
from alibabacloud_openitag20220616 import models as open_itag_models
from evals.tools.itag.sdk.alpha_data_sdk.alpha_data_sdk import AlphaDataSdk
from evals.tools.itag.sdk.alpha_data_sdk import models as alphad_model
from evals.tools.itag.sdk.openitag_sdk.itag_sdk import ItagSdk


class ItagManager(object):
    """
    iTag manager.

    Args:
        tenant_id: Tenant id.
        token: Token.
        employee_id: Employee id.

    Examples:
        >>> from evals.tools import ItagManager
        >>> itag_cfg = dict(tenant_id="xxx", token="xxx", employee_id="xxx")
        >>> itag_manager = ItagManager(**itag_cfg)
        >>> itag_manager.create_dataset()
        >>> itag_manager.create_tag_task()
        >>> itag_manager.get_tag_task_result()

    """

    def __init__(self, tenant_id, token, employee_id, **kwargs):
        self.itag = None
        self.alphad = None
        self._init_itag_client(tenant_id, token, employee_id)

    def _init_itag_client(self, tenant_id, token, employee_id):
        """
        Init iTag client.
        """

        # init iTag sdk
        self.itag = ItagSdk(
            config=open_api_models.Config(
                tenant_id, token, endpoint="itag2.alibaba-inc.com"
            ),
            buc_no=employee_id
        )

        # init AlphaD
        self.alphad = AlphaDataSdk(
            config=open_api_models.Config(
                tenant_id, token, endpoint="alphad.alibaba-inc.com"
            ),
            buc_no=employee_id
        )

        # todo: TBD ...


    def _task_result_parser(self):
        """
        Parse task result.
        """
        pass

    def create_dataset(self):
        """
        Create a dataset from local file.
        """

        # # 创建数据集
        # with open(r"filePath/test.csv", "r", encoding="utf-8") as file:
        #     create_dataset_response = alphad.create_dataset(tenant_id, alphad_model.CreateDatasetRequest(
        #         data_source="LOCAL_FILE",
        #         dataset_name="测试数据集202302271154",
        #         owner_name="张嘉森",
        #         owner_employee_id=employee_id,
        #         file_name="测试数据集202302271154.csv",
        #         file=file,
        #         content_type="multipart/form-data",
        #         secure_level=1,
        #         remark="测试数据集"
        #     ))
        # # 获取创建的数据集ID
        # dataset_id = create_dataset_response.body.result
        # # 等待数据集就绪
        # while True:
        #     dataset_response = alphad.get_dataset(tenant_id, dataset_id)
        #     status = dataset_response.body.result.status
        # if not status:
        #     raise ValueError("dataset status error")
        #
        # if status = "FINISHED":
        #     break
        #
        # time.sleep(5)

    def create_tag_task(self):
        """
        Create a iTag task.
        """
        template_id = ''
        dataset_id = 'prompt测试数据_en'
        dataset_response = self.alphad.get_dataset(tenant_id, str(dataset_id))
        status = dataset_response.body.result.status

        print('>>> dataset status: ', status)

        # create_task_request = open_itag_models.CreateTaskRequest(
        #     body=open_itag_models.CreateTaskDetail(
        #         task_name='测试任务',
        #         template_id=template_id,
        #         task_workflow=[
        #             open_itag_models.CreateTaskDetailTaskWorkflow(
        #                 node_name='MARK'
        #             )
        #         ],
        #         admins=open_itag_models.CreateTaskDetailAdmins(),
        #         assign_config=open_itag_models.TaskAssginConfig(
        #             assign_count=1,
        #             assign_type='FIXED_SIZE'
        #         ),
        #         uuid=str(uuid.uuid4()),
        #         task_template_config=open_itag_models.TaskTemplateConfig(),
        #         dataset_proxy_relations=[
        #             open_itag_models.DatasetProxyConfig(
        #                 source_dataset_id=dataset_response.get("result"),
        #                 dataset_type='LABEL',
        #                 source='ALPHAD'
        #             )
        #         ]
        #     )
        # )
        # create_task_response = self.itag.create_task(tenant_id, create_task_request)
        # print(create_task_response)


    def get_tag_task_result(self):
        """
        Fetch tag task result.
        """
        pass


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
    itag_manager.create_tag_task()

