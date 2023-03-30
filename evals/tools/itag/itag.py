# Copyright (c) Alibaba, Inc. and its affiliates.

from alibabacloud_tea_openapi import models as open_api_models
from alibabacloud_openitag20220616 import models as open_itag_models
from alpha_data_sdk.alpha_data_sdk import AlphaDataSdk
from openitag_sdk.itag_sdk import ItagSdk
from alpha_data_sdk import models as alphad_model


class ItagManager(object):

    def __init__(self):
        # self._init_itag_client()
        pass

    def _init_itag_client(self, tenant_id, token, employee_id):
        """
        Init iTag client.
        """

        # init iTag sdk
        itag = ItagSdk(
            config=open_api_models.Config(
                tenant_id, token, endpoint="itag2.alibaba-inc.com"
            ),
            buc_no=employee_id
        )

        # init AlphaD
        alphad = AlphaDataSdk(
            config=open_api_models.Config(
                tenant_id, token, endpoint="alphad.alibaba-inc.com"
            ),
            buc_no=employee_id
        )

        # todo: TBD ...

    def create_dataset(self):
        """
        Create a dataset from local file.
        """
        pass

    def create_tag_task(self):
        """
        Create a iTag task.
        """
        pass

    def get_tag_task_result(self):
        """
        Fetch tag task result.
        """
        pass
