# Copyright (c) Alibaba, Inc. and its affiliates.


class PredictorMode:
    LOCAL = 'local'
    REMOTE = 'remote'


class PredictorKeys:
    LOCAL_MODEL = 'local_model'


DEFAULT_DASHSCOPE_HTTP_BASE_URL = 'https://int-dashscope.aliyun-inc.com/api/v1/services'


class PredictorEnvs:
    DASHSCOPE_API_KEY = 'DASHSCOPE_API_KEY'
    DASHSCOPE_HTTP_BASE_URL = 'DASHSCOPE_HTTP_BASE_URL'
