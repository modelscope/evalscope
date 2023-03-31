# Copyright (c) Alibaba, Inc. and its affiliates.


class PredictorMode:
    LOCAL = 'local'
    REMOTE = 'remote'


class PredictorKeys:
    LOCAL_MODEL = 'local_model'


class PredictorEnvs:
    # The api key for DashScope, which can be obtained from the DashScope console.
    DASHSCOPE_API_KEY = 'DASHSCOPE_API_KEY'

    # The base url for DashScope, it's necessary when DEBUG_MODE is set to true.
    DEBUG_DASHSCOPE_HTTP_BASE_URL = 'DEBUG_DASHSCOPE_HTTP_BASE_URL'

    # Debug mode, set to 'true' to enable debug mode, otherwise ignore it.
    DEBUG_MODE = 'DEBUG_MODE'
