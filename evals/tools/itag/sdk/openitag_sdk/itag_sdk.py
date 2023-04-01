# -*- coding: utf-8 -*-
# This file is auto-generated, don't edit it. Thanks.
from Tea.exceptions import TeaException
from Tea.core import TeaCore

from alibabacloud_openitag20220616.client import Client as OpenITagClient
from alibabacloud_tea_openapi import models as open_api_models
from alibabacloud_darabonba_string.client import Client as StringClient
from alibabacloud_darabonba_number.client import Client as NumberClient
from alibabacloud_darabonba_array.client import Client as ArrayClient
from darabonba_rsa.client import Client as RSAUtilClient
from alibabacloud_tea_util.client import Client as UtilClient
from alibabacloud_tea_util import models as util_models


class ItagSdk(OpenITagClient):
    _buc_no: str = None

    def __init__(
        self, 
        config: open_api_models.Config,
        buc_no: str,
    ):
        super().__init__(config)
        self._endpoint = self.format_endpoint(self._endpoint)
        self._buc_no = buc_no

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
        构建集团域的请求头
        """
        access_key_id = self.get_access_key_id()
        access_key_secret = self.get_access_key_secret()
        req_timestamp = RSAUtilClient.gen_ts()
        req_timestamp = f'{req_timestamp}'
        req_id = RSAUtilClient.uuid()
        bu_no = self._buc_no
        sign = self.build_token(access_key_id, access_key_secret, bu_no, req_id, req_timestamp)
        request.headers = TeaCore.merge({
            'bucAccountNo': bu_no,
            'alphaqTntInstId': access_key_id,
            'referer': self._endpoint,
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
        构建集团域的请求头
        """
        access_key_id = await self.get_access_key_id_async()
        access_key_secret = await self.get_access_key_secret_async()
        req_timestamp = RSAUtilClient.gen_ts()
        req_timestamp = f'{req_timestamp}'
        req_id = RSAUtilClient.uuid()
        bu_no = self._buc_no
        sign = await self.build_token_async(access_key_id, access_key_secret, bu_no, req_id, req_timestamp)
        request.headers = TeaCore.merge({
            'bucAccountNo': bu_no,
            'alphaqTntInstId': access_key_id,
            'referer': self._endpoint,
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
