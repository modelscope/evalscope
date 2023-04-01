# -*- coding: utf-8 -*-
# This file is auto-generated, don't edit it. Thanks.
import time
import requests

from Tea.request import TeaRequest
from Tea.exceptions import TeaException, UnretryableException
from Tea.core import TeaCore
from typing import BinaryIO, Dict

from alibabacloud_tea_openapi.client import Client as OpenApiClient
from alibabacloud_tea_openapi import models as open_api_models
from alibabacloud_tea_util import models as util_models
from requests_toolbelt import MultipartEncoder

# from alpha_data_sdk import models as alpha_d_models
from ..alpha_data_sdk import models as alpha_d_models
from alibabacloud_tea_util.client import Client as UtilClient
from alibabacloud_tea_fileform.client import Client as FileFormClient
from alibabacloud_tea_fileform import models as file_form_models
from alibabacloud_darabonba_string.client import Client as StringClient
from alibabacloud_darabonba_array.client import Client as ArrayClient
from alibabacloud_darabonba_number.client import Client as NumberClient
from darabonba_rsa.client import Client as RSAUtilClient
from alibabacloud_openapi_util.client import Client as OpenApiUtilClient


class AlphaDataSdk(OpenApiClient):
    """
    *\
    """
    _buc_no: str = None
    _is_log: bool = None

    def __init__(
        self, 
        config: open_api_models.Config,
        buc_no: str,
    ):
        super().__init__(config)
        self._endpoint = self.format_endpoint(self._endpoint)
        self._buc_no = buc_no
        self._is_log = True

    def create_sdk_log(
        self,
        runtime: util_models.RuntimeOptions,
    ) -> alpha_d_models.CreateSdkLogResponse:
        runtime.validate()
        _runtime = {
            # 描述运行时参数
            'timeouted': 'retry',
            'retry': {
                'retryable': runtime.autoretry,
                'maxAttempts': runtime.max_attempts
            },
            'ignoreSSL': runtime.ignore_ssl
        }
        _last_request = None
        _last_exception = None
        _now = time.time()
        _retry_times = 0
        while TeaCore.allow_retry(_runtime.get('retry'), _retry_times, _now):
            if _retry_times > 0:
                _backoff_time = TeaCore.get_backoff_time(_runtime.get('backoff'), _retry_times)
                if _backoff_time > 0:
                    TeaCore.sleep(_backoff_time)
            _retry_times = _retry_times + 1
            try:
                _request = TeaRequest()
                body = {
                    'buc_no': self._buc_no,
                    'endpoint': self._endpoint,
                    'itag_tenant_id': self.get_access_key_id(),
                    'platform_name': 'AlphaD'
                }
                json_obj = UtilClient.to_jsonstring(body)
                # 描述请求相关信息
                _request.protocol = 'https'
                _request.method = 'POST'
                _request.pathname = '/management/v1/create_log'
                _request.headers = {
                    'host': 'itag.alibaba-inc.com',
                    'content-type': 'application/json'
                }
                _request.body = json_obj
                _last_request = _request
                _response = TeaCore.do_action(_request, _runtime)
                # 描述返回相关信息
                return TeaCore.from_map(
                    alpha_d_models.CreateSdkLogResponse(),
                    {
                        'statusCode': _response.status_code
                    }
                )
            except Exception as e:
                if TeaCore.is_retryable(e):
                    _last_exception = e
                    continue
                raise e
        raise UnretryableException(_last_request, _last_exception)

    async def create_sdk_log_async(
        self,
        runtime: util_models.RuntimeOptions,
    ) -> alpha_d_models.CreateSdkLogResponse:
        runtime.validate()
        _runtime = {
            # 描述运行时参数
            'timeouted': 'retry',
            'retry': {
                'retryable': runtime.autoretry,
                'maxAttempts': runtime.max_attempts
            },
            'ignoreSSL': runtime.ignore_ssl
        }
        _last_request = None
        _last_exception = None
        _now = time.time()
        _retry_times = 0
        while TeaCore.allow_retry(_runtime.get('retry'), _retry_times, _now):
            if _retry_times > 0:
                _backoff_time = TeaCore.get_backoff_time(_runtime.get('backoff'), _retry_times)
                if _backoff_time > 0:
                    TeaCore.sleep(_backoff_time)
            _retry_times = _retry_times + 1
            try:
                _request = TeaRequest()
                body = {
                    'buc_no': self._buc_no,
                    'endpoint': self._endpoint,
                    'itag_tenant_id': await self.get_access_key_id_async(),
                    'platform_name': 'AlphaD'
                }
                json_obj = UtilClient.to_jsonstring(body)
                # 描述请求相关信息
                _request.protocol = 'https'
                _request.method = 'POST'
                _request.pathname = '/management/v1/create_log'
                _request.headers = {
                    'host': 'itag.alibaba-inc.com',
                    'content-type': 'application/json'
                }
                _request.body = json_obj
                _last_request = _request
                _response = await TeaCore.async_do_action(_request, _runtime)
                # 描述返回相关信息
                return TeaCore.from_map(
                    alpha_d_models.CreateSdkLogResponse(),
                    {
                        'statusCode': _response.status_code
                    }
                )
            except Exception as e:
                if TeaCore.is_retryable(e):
                    _last_exception = e
                    continue
                raise e
        raise UnretryableException(_last_request, _last_exception)

    def upload_file_from_sts_conf(
        self,
        sts_conf: alpha_d_models.StsConf,
        file_name: str,
        file: BinaryIO,
        content_type: str,
        oss_path: str,
        runtime: util_models.RuntimeOptions,
    ) -> alpha_d_models.UploadFileFromStsConfResponse:
        sts_conf.validate()
        runtime.validate()

        m = MultipartEncoder({
            "OSSAccessKeyId": sts_conf.access_id,
            "policy": sts_conf.policy,
            "signature": sts_conf.signature,
            "success_action_status": "200",
            "key": oss_path,
            "file": file
        })
        res = requests.post(
            sts_conf.host, data=m, headers={
                "Content-Type": m.content_type
            }
        )

        return alpha_d_models.UploadFileFromStsConfResponse(
            status_code=res.status_code
        )

    async def upload_file_from_sts_conf_async(
        self,
        sts_conf: alpha_d_models.StsConf,
        file_name: str,
        file: BinaryIO,
        content_type: str,
        oss_path: str,
        runtime: util_models.RuntimeOptions,
    ) -> alpha_d_models.UploadFileFromStsConfResponse:
        sts_conf.validate()
        runtime.validate()
        _runtime = {
            # 描述运行时参数
            'timeouted': 'retry',
            'retry': {
                'retryable': runtime.autoretry,
                'maxAttempts': runtime.max_attempts
            },
            'ignoreSSL': runtime.ignore_ssl
        }
        _last_request = None
        _last_exception = None
        _now = time.time()
        _retry_times = 0
        while TeaCore.allow_retry(_runtime.get('retry'), _retry_times, _now):
            if _retry_times > 0:
                _backoff_time = TeaCore.get_backoff_time(_runtime.get('backoff'), _retry_times)
                if _backoff_time > 0:
                    TeaCore.sleep(_backoff_time)
            _retry_times = _retry_times + 1
            try:
                _request = TeaRequest()
                # 构建请求参数和文件
                boundary = FileFormClient.get_boundary()
                form = {
                    'OSSAccessKeyId': sts_conf.access_id,
                    'policy': sts_conf.policy,
                    'signature': sts_conf.signature,
                    'success_action_status': '200',
                    'key': oss_path,
                    'file': file_form_models.FileField(
                    filename=file_name,
                    content_type=content_type,
                    content=file
                )
                }
                url_arr = StringClient.split(sts_conf.host, '://', None)
                protocol = ArrayClient.get(url_arr, 0)
                host = ArrayClient.get(url_arr, 1)
                # 描述请求相关信息
                _request.protocol = protocol
                _request.method = 'POST'
                _request.pathname = ''
                _request.headers = {
                    'host': host,
                    'content-type': 'multipart/form-data'
                }
                _request.body = FileFormClient.to_file_form(form, boundary)
                _last_request = _request
                _response = await TeaCore.async_do_action(_request, _runtime)
                # 描述返回相关信息
                if not UtilClient.equal_number(_response.status_code, 200):
                    raise TeaException({
                        'message': f'Reqeust Failed!',
                        'code': f'{_response.status_code}'
                    })
                return TeaCore.from_map(
                    alpha_d_models.UploadFileFromStsConfResponse(),
                    {
                        'statusCode': _response.status_code
                    }
                )
            except Exception as e:
                if TeaCore.is_retryable(e):
                    _last_exception = e
                    continue
                raise e
        raise UnretryableException(_last_request, _last_exception)

    def format_endpoint(
        self,
        endpoint: str,
    ) -> str:
        url_arr = StringClient.split(endpoint, '//', None)
        arr_length = NumberClient.parse_long(f'{ArrayClient.size(url_arr)}')
        index = NumberClient.parse_long(f'{1}')
        max_index = NumberClient.sub(arr_length, index)
        return ArrayClient.get(url_arr, NumberClient.parse_int(f'{max_index}'))

    def build_token(
        self,
        access_key_id: str,
        access_key_secret: str,
        bud_id: str,
        req_id: str,
        ts: str,
    ) -> str:
        line = f'{access_key_id}_{bud_id}_{req_id}_{ts}_tiansuo'
        signature = RSAUtilClient.sha_sign(UtilClient.to_bytes(line), access_key_secret)
        sign = RSAUtilClient.b_encode(signature)
        return UtilClient.to_string(sign)

    async def build_token_async(
        self,
        access_key_id: str,
        access_key_secret: str,
        bud_id: str,
        req_id: str,
        ts: str,
    ) -> str:
        line = f'{access_key_id}_{bud_id}_{req_id}_{ts}_tiansuo'
        signature = RSAUtilClient.sha_sign(UtilClient.to_bytes(line), access_key_secret)
        sign = RSAUtilClient.b_encode(signature)
        return UtilClient.to_string(sign)

    def build_headers(
        self,
        request: open_api_models.OpenApiRequest,
    ) -> open_api_models.OpenApiRequest:
        """
        构建AlphaD集团域的请求头
        """
        access_key_id = self.get_access_key_id()
        access_key_secret = self.get_access_key_secret()
        req_timestamp = RSAUtilClient.gen_ts()
        req_timestamp = f'{req_timestamp}'
        req_id = RSAUtilClient.uuid()
        bu_no = self._buc_no
        sign = self.build_token(access_key_id, access_key_secret, bu_no, req_id, req_timestamp)
        if not self._is_log:
            log_runtime = util_models.RuntimeOptions()
            self.create_sdk_log(log_runtime)
            self._is_log = True
        request.headers = TeaCore.merge({
            'bucAccountNo': bu_no,
            'referer': self._endpoint,
            'alpha-tnt': access_key_id,
            'ReqSrc': 'aITAG',
            'reqTimestamp': req_timestamp,
            'reqId': req_id,
            'Content-type': 'Application/json;charset=UTF-8',
            'reqSourceApp': 'tiansuo',
            'token': sign
        }, request.headers)
        return request

    async def build_headers_async(
        self,
        request: open_api_models.OpenApiRequest,
    ) -> open_api_models.OpenApiRequest:
        """
        构建AlphaD集团域的请求头
        """
        access_key_id = await self.get_access_key_id_async()
        access_key_secret = await self.get_access_key_secret_async()
        req_timestamp = RSAUtilClient.gen_ts()
        req_timestamp = f'{req_timestamp}'
        req_id = RSAUtilClient.uuid()
        bu_no = self._buc_no
        sign = await self.build_token_async(access_key_id, access_key_secret, bu_no, req_id, req_timestamp)
        if not self._is_log:
            log_runtime = util_models.RuntimeOptions()
            await self.create_sdk_log_async(log_runtime)
            self._is_log = True
        request.headers = TeaCore.merge({
            'bucAccountNo': bu_no,
            'referer': self._endpoint,
            'alpha-tnt': access_key_id,
            'ReqSrc': 'aITAG',
            'reqTimestamp': req_timestamp,
            'reqId': req_id,
            'Content-type': 'Application/json;charset=UTF-8',
            'reqSourceApp': 'tiansuo',
            'token': sign
        }, request.headers)
        return request

    def call_api(
        self,
        params: open_api_models.Params,
        request: open_api_models.OpenApiRequest,
        runtime: util_models.RuntimeOptions,
    ) -> dict:
        if UtilClient.is_unset(params):
            raise TeaException({
                'code': 'ParameterMissing',
                'message': "'params' can not be unset"
            })
        request = self.build_headers(request)
        if UtilClient.is_unset(self._signature_algorithm) or not UtilClient.equal_string(self._signature_algorithm, 'v2'):
            return self.do_request(params, request, runtime)
        elif UtilClient.equal_string(params.style, 'ROA') and UtilClient.equal_string(params.req_body_type, 'json'):
            return self.do_roarequest(params.action, params.version, params.protocol, params.method, params.auth_type, params.pathname, params.body_type, request, runtime)
        elif UtilClient.equal_string(params.style, 'ROA'):
            return self.do_roarequest_with_form(params.action, params.version, params.protocol, params.method, params.auth_type, params.pathname, params.body_type, request, runtime)
        else:
            return self.do_rpcrequest(params.action, params.version, params.protocol, params.method, params.auth_type, params.body_type, request, runtime)

    async def call_api_async(
        self,
        params: open_api_models.Params,
        request: open_api_models.OpenApiRequest,
        runtime: util_models.RuntimeOptions,
    ) -> dict:
        if UtilClient.is_unset(params):
            raise TeaException({
                'code': 'ParameterMissing',
                'message': "'params' can not be unset"
            })
        request = await self.build_headers_async(request)
        if UtilClient.is_unset(self._signature_algorithm) or not UtilClient.equal_string(self._signature_algorithm, 'v2'):
            return await self.do_request_async(params, request, runtime)
        elif UtilClient.equal_string(params.style, 'ROA') and UtilClient.equal_string(params.req_body_type, 'json'):
            return await self.do_roarequest_async(params.action, params.version, params.protocol, params.method, params.auth_type, params.pathname, params.body_type, request, runtime)
        elif UtilClient.equal_string(params.style, 'ROA'):
            return await self.do_roarequest_with_form_async(params.action, params.version, params.protocol, params.method, params.auth_type, params.pathname, params.body_type, request, runtime)
        else:
            return await self.do_rpcrequest_async(params.action, params.version, params.protocol, params.method, params.auth_type, params.body_type, request, runtime)

    def list_datasets(
        self,
        tenant_id: str,
        request: alpha_d_models.ListDatasetsRequest,
    ) -> alpha_d_models.ListDatasetsResponse:
        runtime = util_models.RuntimeOptions()
        headers = {}
        return self.list_datasets_with_options(tenant_id, request, headers, runtime)

    async def list_datasets_async(
        self,
        tenant_id: str,
        request: alpha_d_models.ListDatasetsRequest,
    ) -> alpha_d_models.ListDatasetsResponse:
        runtime = util_models.RuntimeOptions()
        headers = {}
        return await self.list_datasets_with_options_async(tenant_id, request, headers, runtime)

    def list_datasets_with_options(
        self,
        tenant_id: str,
        request: alpha_d_models.ListDatasetsRequest,
        headers: Dict[str, str],
        runtime: util_models.RuntimeOptions,
    ) -> alpha_d_models.ListDatasetsResponse:
        body = {}
        if not UtilClient.is_unset(request.source):
            body['source'] = request.source
        if not UtilClient.is_unset(request.collection_mix_info):
            body['collectionMixInfo'] = request.collection_mix_info
        if not UtilClient.is_unset(request.name):
            body['name'] = request.name
        if not UtilClient.is_unset(request.page_num):
            body['pageNum'] = request.page_num
        if not UtilClient.is_unset(request.page_size):
            body['pageSize'] = request.page_size
        if not UtilClient.is_unset(request.set_type):
            body['setType'] = request.set_type
        if not UtilClient.is_unset(request.shared_type):
            body['sharedType'] = request.shared_type
        if not UtilClient.is_unset(request.tag):
            body['tag'] = request.tag
        if not UtilClient.is_unset(request.contain_deleted):
            body['containDeleted'] = request.contain_deleted
        if not UtilClient.is_unset(request.creator_id):
            body['creatorId'] = request.creator_id
        req = open_api_models.OpenApiRequest(
            headers=headers,
            body=body
        )
        params = open_api_models.Params(
            action='listDatasets',
            version='2023-02-21',
            protocol='HTTPS',
            pathname=f'/api/adf/datamng/queryCollectionByPage',
            method='POST',
            auth_type='AK',
            style='ROA',
            req_body_type='json',
            body_type='json'
        )
        return TeaCore.from_map(
            alpha_d_models.ListDatasetsResponse(),
            self.call_api(params, req, runtime)
        )

    async def list_datasets_with_options_async(
        self,
        tenant_id: str,
        request: alpha_d_models.ListDatasetsRequest,
        headers: Dict[str, str],
        runtime: util_models.RuntimeOptions,
    ) -> alpha_d_models.ListDatasetsResponse:
        body = {}
        if not UtilClient.is_unset(request.source):
            body['source'] = request.source
        if not UtilClient.is_unset(request.collection_mix_info):
            body['collectionMixInfo'] = request.collection_mix_info
        if not UtilClient.is_unset(request.name):
            body['name'] = request.name
        if not UtilClient.is_unset(request.page_num):
            body['pageNum'] = request.page_num
        if not UtilClient.is_unset(request.page_size):
            body['pageSize'] = request.page_size
        if not UtilClient.is_unset(request.set_type):
            body['setType'] = request.set_type
        if not UtilClient.is_unset(request.shared_type):
            body['sharedType'] = request.shared_type
        if not UtilClient.is_unset(request.tag):
            body['tag'] = request.tag
        if not UtilClient.is_unset(request.contain_deleted):
            body['containDeleted'] = request.contain_deleted
        if not UtilClient.is_unset(request.creator_id):
            body['creatorId'] = request.creator_id
        req = open_api_models.OpenApiRequest(
            headers=headers,
            body=body
        )
        params = open_api_models.Params(
            action='listDatasets',
            version='2023-02-21',
            protocol='HTTPS',
            pathname=f'/api/adf/datamng/queryCollectionByPage',
            method='POST',
            auth_type='AK',
            style='ROA',
            req_body_type='json',
            body_type='json'
        )
        return TeaCore.from_map(
            alpha_d_models.ListDatasetsResponse(),
            await self.call_api_async(params, req, runtime)
        )

    def get_dataset(
        self,
        tenant_id: str,
        dataset_id: str,
    ) -> alpha_d_models.GetDatasetResponse:
        runtime = util_models.RuntimeOptions()
        headers = {}
        return self.get_dataset_with_options(tenant_id, dataset_id, headers, runtime)

    async def get_dataset_async(
        self,
        tenant_id: str,
        dataset_id: str,
    ) -> alpha_d_models.GetDatasetResponse:
        runtime = util_models.RuntimeOptions()
        headers = {}
        return await self.get_dataset_with_options_async(tenant_id, dataset_id, headers, runtime)

    def get_dataset_with_options(
        self,
        tenant_id: str,
        dataset_id: str,
        headers: Dict[str, str],
        runtime: util_models.RuntimeOptions,
    ) -> alpha_d_models.GetDatasetResponse:
        req = open_api_models.OpenApiRequest(
            headers=headers
        )
        params = open_api_models.Params(
            action='getDataset',
            version='2023-02-21',
            protocol='HTTPS',
            pathname=f'/adf/datamng/queryCollectionById/{OpenApiUtilClient.get_encode_param(dataset_id)}',
            method='POST',
            auth_type='AK',
            style='ROA',
            req_body_type='json',
            body_type='json'
        )
        return TeaCore.from_map(
            alpha_d_models.GetDatasetResponse(),
            self.call_api(params, req, runtime)
        )

    async def get_dataset_with_options_async(
        self,
        tenant_id: str,
        dataset_id: str,
        headers: Dict[str, str],
        runtime: util_models.RuntimeOptions,
    ) -> alpha_d_models.GetDatasetResponse:
        req = open_api_models.OpenApiRequest(
            headers=headers
        )
        params = open_api_models.Params(
            action='getDataset',
            version='2023-02-21',
            protocol='HTTPS',
            pathname=f'/adf/datamng/queryCollectionById/{OpenApiUtilClient.get_encode_param(dataset_id)}',
            method='POST',
            auth_type='AK',
            style='ROA',
            req_body_type='json',
            body_type='json'
        )
        return TeaCore.from_map(
            alpha_d_models.GetDatasetResponse(),
            await self.call_api_async(params, req, runtime)
        )

    def get_init_collection_schema(
        self,
        tenant_id: str,
        data_source: str,
        oss_conf: alpha_d_models.OssConf,
        oss_path: str,
    ) -> alpha_d_models.GetInitCollectionSchemaResponse:
        runtime = util_models.RuntimeOptions()
        headers = {}
        return self.get_init_collection_schemat_with_options(tenant_id, data_source, oss_conf, oss_path, headers, runtime)

    async def get_init_collection_schema_async(
        self,
        tenant_id: str,
        data_source: str,
        oss_conf: alpha_d_models.OssConf,
        oss_path: str,
    ) -> alpha_d_models.GetInitCollectionSchemaResponse:
        runtime = util_models.RuntimeOptions()
        headers = {}
        return await self.get_init_collection_schemat_with_options_async(tenant_id, data_source, oss_conf, oss_path, headers, runtime)

    def get_init_collection_schemat_with_options(
        self,
        tenant_id: str,
        data_source: str,
        oss_conf: alpha_d_models.OssConf,
        oss_path: str,
        headers: Dict[str, str],
        runtime: util_models.RuntimeOptions,
    ) -> alpha_d_models.GetInitCollectionSchemaResponse:
        oss_config = {}
        if UtilClient.equal_string(data_source, 'LOCAL_FILE'):
            oss_config['dataSource'] = 'LOCAL_FILE'
            oss_config['ossPath'] = oss_path
        if UtilClient.equal_string(data_source, 'LOCAL_DIR'):
            oss_config['dataSource'] = 'LOCAL_DIR'
            oss_config['ossPath'] = oss_path
        if UtilClient.equal_string(data_source, 'OSS_FILE'):
            oss_config['dataSource'] = 'OSS_FILE'
            oss_config['ossAk'] = oss_conf.key
            oss_config['ossAs'] = oss_conf.token
            oss_config['ossBucket'] = oss_conf.bucket
            oss_config['ossEndpoint'] = oss_conf.endpoint
            oss_config['ossPath'] = oss_conf.path
        body = {
            'collectionVO': {
                'dataSource': data_source,
                'id': None,
                'periodConfig': {},
                'remark': '',
                'setType': 'ALPHAD_TABLE'
            },
            'metaInfo': {
                'dataSource': data_source,
                'ossConfig': oss_config
            },
            'schemaVOList': [],
            'source': '__itag'
        }

        req = open_api_models.OpenApiRequest(
            headers=headers,
            body=body
        )
        params = open_api_models.Params(
            action='getInitCollectionSchema',
            version='2023-02-21',
            protocol='HTTPS',
            pathname=f'/api/adf/datamng/initCollectionSchema',
            method='POST',
            auth_type='AK',
            style='ROA',
            req_body_type='json',
            body_type='json'
        )
        return TeaCore.from_map(
            alpha_d_models.GetInitCollectionSchemaResponse(),
            self.call_api(params, req, runtime)
        )

    async def get_init_collection_schemat_with_options_async(
        self,
        tenant_id: str,
        data_source: str,
        oss_conf: alpha_d_models.OssConf,
        oss_path: str,
        headers: Dict[str, str],
        runtime: util_models.RuntimeOptions,
    ) -> alpha_d_models.GetInitCollectionSchemaResponse:
        oss_config = {}
        if UtilClient.equal_string(data_source, 'LOCAL_FILE'):
            oss_config['dataSource'] = 'LOCAL_FILE'
            oss_config['ossPath'] = oss_path
        if UtilClient.equal_string(data_source, 'LOCAL_DIR'):
            oss_config['dataSource'] = 'LOCAL_DIR'
            oss_config['ossPath'] = oss_path
        if UtilClient.equal_string(data_source, 'OSS_FILE'):
            oss_config['dataSource'] = 'OSS_FILE'
            oss_config['ossAk'] = oss_conf.key
            oss_config['ossAs'] = oss_conf.token
            oss_config['ossBucket'] = oss_conf.bucket
            oss_config['ossEndpoint'] = oss_conf.endpoint
            oss_config['ossPath'] = oss_conf.path
        body = {
            'collectionVO': {
                'dataSource': data_source,
                'id': None,
                'periodConfig': {},
                'remark': '',
                'setType': 'ALPHAD_TABLE'
            },
            'metaInfo': {
                'dataSource': data_source,
                'ossConfig': oss_config
            },
            'schemaVOList': {},
            'source': '__itag'
        }
        req = open_api_models.OpenApiRequest(
            headers=headers,
            body=body
        )
        params = open_api_models.Params(
            action='getInitCollectionSchema',
            version='2023-02-21',
            protocol='HTTPS',
            pathname=f'/api/adf/datamng/initCollectionSchema',
            method='POST',
            auth_type='AK',
            style='ROA',
            req_body_type='json',
            body_type='json'
        )
        return TeaCore.from_map(
            alpha_d_models.GetInitCollectionSchemaResponse(),
            await self.call_api_async(params, req, runtime)
        )

    def get_oss_post_signature(
        self,
        tenant_id: str,
    ) -> alpha_d_models.GetOssPostSignatureResponse:
        runtime = util_models.RuntimeOptions()
        headers = {}
        return self.get_oss_post_signature_with_options(tenant_id, headers, runtime)

    async def get_oss_post_signature_async(
        self,
        tenant_id: str,
    ) -> alpha_d_models.GetOssPostSignatureResponse:
        runtime = util_models.RuntimeOptions()
        headers = {}
        return await self.get_oss_post_signature_with_options_async(tenant_id, headers, runtime)

    def get_oss_post_signature_with_options(
        self,
        tenant_id: str,
        headers: Dict[str, str],
        runtime: util_models.RuntimeOptions,
    ) -> alpha_d_models.GetOssPostSignatureResponse:
        body = {
            'dataSource': 'LOCAL_FILE',
            'source': '__itag'
        }
        req = open_api_models.OpenApiRequest(
            headers=headers,
            body=body
        )
        params = open_api_models.Params(
            action='getOssPostSignature',
            version='2023-02-21',
            protocol='HTTPS',
            pathname=f'/adf/datamng/getOssPostSignature',
            method='POST',
            auth_type='AK',
            style='ROA',
            req_body_type='json',
            body_type='json'
        )
        return TeaCore.from_map(
            alpha_d_models.GetOssPostSignatureResponse(),
            self.call_api(params, req, runtime)
        )

    async def get_oss_post_signature_with_options_async(
        self,
        tenant_id: str,
        headers: Dict[str, str],
        runtime: util_models.RuntimeOptions,
    ) -> alpha_d_models.GetOssPostSignatureResponse:
        body = {
            'dataSource': 'LOCAL_FILE',
            'source': '__itag'
        }
        req = open_api_models.OpenApiRequest(
            headers=headers,
            body=body
        )
        params = open_api_models.Params(
            action='getOssPostSignature',
            version='2023-02-21',
            protocol='HTTPS',
            pathname=f'/adf/datamng/getOssPostSignature',
            method='POST',
            auth_type='AK',
            style='ROA',
            req_body_type='json',
            body_type='json'
        )
        return TeaCore.from_map(
            alpha_d_models.GetOssPostSignatureResponse(),
            await self.call_api_async(params, req, runtime)
        )

    def create_dataset(
        self,
        tenant_id: str,
        create_dataset_request: alpha_d_models.CreateDatasetRequest,
    ) -> alpha_d_models.CreateDatasetResponse:
        runtime = util_models.RuntimeOptions()
        headers = {}
        return self.create_dataset_with_options(tenant_id, create_dataset_request, headers, runtime)

    async def create_dataset_async(
        self,
        tenant_id: str,
        create_dataset_request: alpha_d_models.CreateDatasetRequest,
    ) -> alpha_d_models.CreateDatasetResponse:
        runtime = util_models.RuntimeOptions()
        headers = {}
        return await self.create_dataset_with_options_async(tenant_id, create_dataset_request, headers, runtime)

    def create_dataset_with_options(
        self,
        tenant_id: str,
        create_dataset_request: alpha_d_models.CreateDatasetRequest,
        headers: Dict[str, str],
        runtime: util_models.RuntimeOptions,
    ) -> alpha_d_models.CreateDatasetResponse:
        oss_config = {}
        oss_path = ''
        if UtilClient.equal_string(create_dataset_request.data_source, 'LOCAL_FILE'):
            local_file_oss_post_signature_response = self.get_oss_post_signature(tenant_id)
            local_file_sts_conf = local_file_oss_post_signature_response.body.result
            local_file_upload_runtime = util_models.RuntimeOptions(read_timeout=20000, connect_timeout=20000)
            oss_path = f'{local_file_sts_conf.dir}{create_dataset_request.file_name}'
            oss_config['dataSource'] = 'LOCAL_FILE'
            oss_config['ossPath'] = oss_path
            self.upload_file_from_sts_conf(local_file_sts_conf, create_dataset_request.file_name, create_dataset_request.file, create_dataset_request.content_type, oss_path, local_file_upload_runtime)
        if UtilClient.equal_string(create_dataset_request.data_source, 'OSS_FILE'):
            oss_path = create_dataset_request.oss_conf.path
            oss_config['dataSource'] = 'OSS_FILE'
            oss_config['ossAk'] = create_dataset_request.oss_conf.key
            oss_config['ossAs'] = create_dataset_request.oss_conf.token
            oss_config['ossBucket'] = create_dataset_request.oss_conf.bucket
            oss_config['ossEndpoint'] = create_dataset_request.oss_conf.endpoint
            oss_config['ossPath'] = oss_path
        if UtilClient.equal_string(create_dataset_request.data_source, 'OSS_DIR'):
            oss_path = create_dataset_request.oss_conf.path
            oss_config['dataSource'] = 'OSS_DIR'
            oss_config['ossAk'] = create_dataset_request.oss_conf.key
            oss_config['ossAs'] = create_dataset_request.oss_conf.token
            oss_config['ossBucket'] = create_dataset_request.oss_conf.bucket
            oss_config['ossEndpoint'] = create_dataset_request.oss_conf.endpoint
            oss_config['ossPath'] = oss_path
        init_collection_schema_response = self.get_init_collection_schema(tenant_id, create_dataset_request.data_source, create_dataset_request.oss_conf, oss_path)
        schema_response_map = UtilClient.to_map(init_collection_schema_response.body)
        body = {
            'collectionVO': {
                'dataSource': create_dataset_request.data_source,
                'id': None,
                'name': create_dataset_request.dataset_name,
                'secureLevel': create_dataset_request.secure_level,
                'ownerList': [
                    {
                        'nickName': create_dataset_request.owner_name,
                        'staffNo': create_dataset_request.owner_employee_id,
                        'isCreator': True
                    }
                ],
                'remark': create_dataset_request.remark,
                'setType': 'ALPHAD_TABLE',
                'periodConfig': {},
                'mode': 'PROTECTED',
                'tntInstId': tenant_id,
                'sharedMode': 'USABLE',
                'creatorId': create_dataset_request.owner_employee_id,
                'bizType': 2
            },
            'metaInfo': {
                'dataSource': create_dataset_request.data_source,
                'ossConfig': oss_config
            },
            'schemaVOList': schema_response_map.get('result')
        }
        req = open_api_models.OpenApiRequest(
            headers=headers,
            body=body
        )
        params = open_api_models.Params(
            action='CreateDataset',
            version='2023-02-21',
            protocol='HTTPS',
            pathname=f'/api/adf/datamng/create',
            method='POST',
            auth_type='AK',
            style='ROA',
            req_body_type='json',
            body_type='json'
        )
        return TeaCore.from_map(
            alpha_d_models.CreateDatasetResponse(),
            self.call_api(params, req, runtime)
        )

    async def create_dataset_with_options_async(
        self,
        tenant_id: str,
        create_dataset_request: alpha_d_models.CreateDatasetRequest,
        headers: Dict[str, str],
        runtime: util_models.RuntimeOptions,
    ) -> alpha_d_models.CreateDatasetResponse:
        oss_config = {}
        oss_path = ''
        if UtilClient.equal_string(create_dataset_request.data_source, 'LOCAL_FILE'):
            local_file_oss_post_signature_response = await self.get_oss_post_signature_async(tenant_id)
            local_file_sts_conf = local_file_oss_post_signature_response.body.result
            local_file_upload_runtime = util_models.RuntimeOptions()
            oss_path = f'{local_file_sts_conf.dir}{create_dataset_request.file_name}'
            oss_config['dataSource'] = 'LOCAL_FILE'
            oss_config['ossPath'] = oss_path
            await self.upload_file_from_sts_conf_async(local_file_sts_conf, create_dataset_request.file_name, create_dataset_request.file, create_dataset_request.content_type, oss_path, local_file_upload_runtime)
        if UtilClient.equal_string(create_dataset_request.data_source, 'OSS_FILE') or UtilClient.equal_string(create_dataset_request.data_source, 'OSS_DIR'):
            oss_path = create_dataset_request.oss_conf.path
            oss_config['dataSource'] = 'OSS_FILE'
            oss_config['ossAk'] = create_dataset_request.oss_conf.key
            oss_config['ossAs'] = create_dataset_request.oss_conf.token
            oss_config['ossBucket'] = create_dataset_request.oss_conf.bucket
            oss_config['ossEndpoint'] = create_dataset_request.oss_conf.endpoint
            oss_config['ossPath'] = oss_path
        init_collection_schema_response = await self.get_init_collection_schema_async(tenant_id, create_dataset_request.data_source, create_dataset_request.oss_conf, oss_path)
        schema_response_map = UtilClient.to_map(init_collection_schema_response.body)
        body = {
            'collectionVO': {
                'dataSource': create_dataset_request.data_source,
                'id': None,
                'name': create_dataset_request.dataset_name,
                'secureLevel': create_dataset_request.secure_level,
                'ownerList': [
                    {
                        'nickName': create_dataset_request.owner_name,
                        'staffNo': create_dataset_request.owner_employee_id,
                        'isCreator': True
                    }
                ],
                'remark': create_dataset_request.remark,
                'setType': 'ALPHAD_TABLE',
                'periodConfig': {},
                'mode': 'PROTECTED',
                'tntInstId': tenant_id,
                'sharedMode': 'USABLE',
                'creatorId': create_dataset_request.owner_employee_id,
                'bizType': 2
            },
            'metaInfo': {
                'dataSource': create_dataset_request.data_source,
                'ossConfig': oss_config
            },
            'schemaVOList': schema_response_map.get('result')
        }
        req = open_api_models.OpenApiRequest(
            headers=headers,
            body=body
        )
        params = open_api_models.Params(
            action='CreateDataset',
            version='2023-02-21',
            protocol='HTTPS',
            pathname=f'/api/adf/datamng/create',
            method='POST',
            auth_type='AK',
            style='ROA',
            req_body_type='json',
            body_type='json'
        )
        return TeaCore.from_map(
            alpha_d_models.CreateDatasetResponse(),
            await self.call_api_async(params, req, runtime)
        )
