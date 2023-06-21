# Copyright (c) Alibaba, Inc. and its affiliates.

import os.path
import time
import uuid
from typing import Union

import pandas as pd
from alibabacloud_openitag20220616 import models as open_itag_models
from alibabacloud_tea_openapi import models as open_api_models
from evals.constants import ItagEnvs
from evals.utils.logger import get_logger
from ITagSDK import Config, ItagSdk, itag_models
from ITagSDK.alpha_data_sdk import models as alphad_model
from ITagSDK.alpha_data_sdk.alpha_data_sdk import AlphaDataSdk

logger = get_logger()


class ItagManager(object):
    """
    iTag manager.

    Args:
        tenant_id: Tenant id.
        token: Token.
        employee_id: Employee id.

    Examples:
        >>> from evals.tools import ItagManager
        >>> itag_manager = ItagManager(tenant_id="your-tenant-id", token="your-token", employee_id="your-employee-id")
        >>> task_resp = itag_manager.process(dataset_path="your-dataset.csv", dataset_name='your-dataset-name',
                                            template_id="your-template-id", task_name="your-task-name")
        >>> df_res = itag_manager.get_tag_task_result(task_id=task_resp.get('TaskId'))
    """

    def __init__(self, tenant_id, token, employee_id):
        self._tenant_id = tenant_id
        self._token = token
        self._employee_id = employee_id

        self._endpoint = os.environ.get(ItagEnvs.ITAG_INTERNAL_ENDPOINT, None)
        if not self._endpoint:
            raise ValueError('env ITAG_INTERNAL_ENDPOINT is not set')

        self._alphad_endpoint = os.environ.get('ALPHAD_INTERNAL_ENDPOINT',
                                               None)
        if not self._alphad_endpoint:
            raise ValueError('env ALPHAD_INTERNAL_ENDPOINT is not set')

        self._itag = None
        self._alphad = None
        self._init_itag_client(tenant_id, token, employee_id)

    def _init_itag_client(self, tenant_id, token, employee_id):
        """
        Init iTag client.
        """

        # init iTag sdk
        self._itag = ItagSdk(
            config=Config(tenant_id, token, endpoint=self._endpoint),
            buc_no=employee_id)

        # init AlphaD
        self._alphad = AlphaDataSdk(
            config=open_api_models.Config(
                tenant_id, token, endpoint=self._alphad_endpoint),
            buc_no=employee_id)

    def _task_result_parser(self):
        """
        Parse task result.
        """
        pass

    def create_dataset(self, dataset_file_path, dataset_name: str) -> str:
        """
        Create a dataset from local file.

        Args:
            dataset_file_path (str): Dataset file path. Should be a csv file which is aligned with template on fields.
            dataset_name (str): Dataset name.

        Returns:
            Dataset id. (str)
        """

        # Create dataset from local file
        with open(dataset_file_path, 'rb') as f:
            create_dataset_response = self._alphad.create_dataset(
                self._tenant_id,
                alphad_model.CreateDatasetRequest(
                    data_source='LOCAL_FILE',
                    dataset_name=dataset_name,
                    owner_name='User_' + str(self._employee_id),
                    owner_employee_id=self._employee_id,
                    file_name=os.path.basename(dataset_file_path),
                    file=f,
                    content_type='multipart/form-data',
                    secure_level=1,
                    remark='AlphaD dataset'))
        logger.info(f'>>create dataset resp: {create_dataset_response}')

        # Get created dataset id
        dataset_id = create_dataset_response.body.result
        logger.info(f'>>create dataset id: {dataset_id}')

        # To wait for dataset creation
        while True:
            dataset_response = self._alphad.get_dataset(
                self._tenant_id, dataset_id)
            status = dataset_response.body.result.status
            if not status:
                raise ValueError('dataset status error')

            if status == 'FINISHED':
                break

            time.sleep(5)

        return str(dataset_id)

    def get_dataset_list(self):
        """
        Get dataset list on the iTag.
        """
        datasets_list_resp = self._alphad.list_datasets(
            self._tenant_id,
            alphad_model.ListDatasetsRequest(
                page_size=10,
                page_num=1,
                contain_deleted=False,
                source='_itag',
                set_type='ALPHAD_TABLE',
                shared_type='USABLE',
                creator_id=self._employee_id))

        return datasets_list_resp

    def get_dataset_info(self, dataset_id: Union[str, int]):
        """
        Get dataset info.
        """
        dataset_id = str(dataset_id)
        dataset_info = self._alphad.get_dataset(self._tenant_id, dataset_id)

        return dataset_info

    def list_users(self, page_size: int = 10, page_number: int = 1) -> dict:
        """
        List iTag users in given tenant scope.
        :param page_size:
        :param page_number:
        :return: users dict.
            Example: {'Code': 0, 'Message': 'success', 'PageNumber': 1, 'PageSize': 10, 'RequestId': 'xxx',
            'Success': True, 'TotalCount': 835, 'TotalPage': 84, 'Users': [{'AccountNo': 'xxx', 'AccountType': 'BUC',
            'Role': 'ADMIN', 'UserId': 12345, 'UserName': 'xiaoming'}, ...]}
        """
        request = itag_models.ListUsersRequest(
            page_size=page_size, page_number=page_number)
        resp = self._itag.list_users(self._tenant_id, request)

        return resp.body.to_map()

    def list_tasks(self, page_size: int = 10, page_number: int = 1) -> dict:
        request = itag_models.ListTasksRequest(
            page_size=page_size, page_number=page_number)
        resp = self._itag.list_tasks(self._tenant_id, request)
        resp = resp.body.to_map()
        if resp['Code'] != 0:
            raise ValueError(f'list tasks failed, resp: {resp}')

        return resp['Tasks']

    def create_tag_task(self, task_name: str, dataset_id: Union[str, int],
                        template_id: str):
        """
        Create a iTag task.
        """
        dataset_info = self.get_dataset_info(dataset_id)
        status = dataset_info.body.result.status
        logger.info(f'Current status of dataset: {status}')

        create_task_request = open_itag_models.CreateTaskRequest(
            body=open_itag_models.CreateTaskDetail(
                task_name=task_name,
                template_id=template_id,
                task_workflow=[
                    open_itag_models.CreateTaskDetailTaskWorkflow(
                        node_name='MARK')
                ],
                admins=open_itag_models.CreateTaskDetailAdmins(),
                assign_config=open_itag_models.TaskAssginConfig(
                    assign_count=1, assign_type='FIXED_SIZE'),
                uuid=str(uuid.uuid4()),
                task_template_config=open_itag_models.TaskTemplateConfig(),
                dataset_proxy_relations=[
                    open_itag_models.DatasetProxyConfig(
                        # source_dataset_id=dataset_info.get("result"),
                        source_dataset_id=str(dataset_id),
                        dataset_type='LABEL',
                        source='ALPHAD')
                ]))
        create_task_response = self._itag.create_task(self._tenant_id,
                                                      create_task_request)

        return create_task_response

    def get_tag_task_result(self, task_id: str) -> pd.DataFrame:
        """
        Fetch tag task result.

        Args:
            task_id (str): Task id.

        Returns:
            Task result. (pd.DataFrame)
        """
        anno_response = self._itag.export_annotations(
            self._tenant_id, task_id,
            open_itag_models.ExportAnnotationsRequest())
        job_id = anno_response.body.flow_job.job_id
        logger.info(f'>job_id: {job_id}')

        job_request = open_itag_models.GetJobRequest(
            job_type='DOWNLOWD_MARKRESULT_FLOW')

        max_iter = 30
        idx = 0
        df_result = None
        while True:
            if idx >= max_iter:
                break
            job_response = self._itag.get_job(
                self._tenant_id, job_id, request=job_request)
            status = job_response.body.job.status

            if status not in ['init', 'running', 'succ']:
                raise ValueError(job_response.body.message)

            if status == 'succ':
                body_dict = job_response.body.to_map()
                result_link = body_dict.get('Job',
                                            {}).get('JobResult',
                                                    {}).get('ResultLink', '')
                df_result = pd.read_csv(result_link)
                break

            time.sleep(1)
            idx += 1

        if idx >= max_iter and not df_result:
            logger.error('Timeout to get iTag task result')

        return df_result

    def list_subtasks(self,
                      task_id: str,
                      page_size: int = 10,
                      page_number: int = 1):
        # TODO: list subtasks
        import json
        request = itag_models.ListSubtasksRequest(
            page_size=page_size, page_number=page_number)

        sub_task_resp = self._itag.list_subtasks(
            tenant_id=self._tenant_id, task_id=task_id, request=request)
        sub_task_resp = sub_task_resp.body.to_map()
        res_list = sub_task_resp.get('Subtasks', [])

        return res_list

    def process(self, dataset_path: str, dataset_name: str, template_id: str,
                task_name: str) -> dict:
        """
        Entry pipeline for creating the iTag task from local csv file.

        Args:
            dataset_path (str): Dataset file path. Should be a csv file which is aligned with template on fields.
            dataset_name (str): Specify a dataset name.
            template_id (str): Template id.
            task_name (str): Specify a task name.

        Returns:
            Task response body from iTag. (dict)
        """

        # Create dataset from local file
        dataset_id = self.create_dataset(dataset_path, dataset_name)

        # Create iTag task
        create_task_response = self.create_tag_task(
            task_name=task_name,
            dataset_id=dataset_id,
            template_id=template_id)
        logger.info(f'>>create task resp: {create_task_response}')
        return create_task_response.body.to_map()
