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


class EvalTaskConfig:
    TASK_NAME = 'task_name'
    TASK_ID = 'id'
    SAMPLES = 'samples'
    SCORING_MODEL = 'scoring_model'
    PREDICTOR = 'predictor'
    CLASS_REF = 'ref'
    CLASS_ARGS = 'args'
    ARGS_MODEL = 'model'
    ARGS_MAX_LEN = 'max_length'
    ARGS_TOP_K = 'top_k'


class ScoringModel:
    GENERATION_EVAL = 'generation_eval'
    CLASSIFICATION_EVAL = 'classification_eval'
    MATCH_EVAL = 'match_eval'
    INCLUDES_EVAL = 'includes_eval'
    SIMILARITY_EVAL = 'similarity_eval'
    UNIT_TEST_EVAL = 'unit_test_eval'
