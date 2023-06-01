# Copyright (c) Alibaba, Inc. and its affiliates.
from enum import Enum

DEFAULT_WORK_DIR = "~/maas_evals"


class TaskEnvs:
    # The cache root dir for tasks
    WORK_DIR = 'WORK_DIR'


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


class ItagEnvs:

    ITAG_INTERNAL_ENDPOINT = 'ITAG_INTERNAL_ENDPOINT'

    ALPHAD_INTERNAL_ENDPOINT = 'ALPHAD_INTERNAL_ENDPOINT'


class EvalTaskConfig:
    TASK_NAME = 'task_name'
    TASK_ID = 'id'
    SAMPLES = 'samples'
    EVALUATOR = 'evaluator'
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


class DumpMode:
    OVERWRITE = 'overwrite'
    APPEND = 'append'


class MetricsConstant:
    EPSILON = float(1e-6)
    INVALID_VALUE = -9999999
    ROUGE_KEYS = [
        'rouge-1-r', 'rouge-1-p', 'rouge-1-f',
        'rouge-2-r', 'rouge-2-p', 'rouge-2-f',
        'rouge-l-r', 'rouge-l-p', 'rouge-l-f',
    ]


class MetricMembers(Enum):

    # Math accuracy metric
    MATH_ACCURACY = 'math_accuracy'

    # Code pass@k metric
    CODE_PASS_K = 'code_pass_k'

    # Code rouge metric
    ROUGE = 'rouge'

    # ELO rating system for pairwise comparison
    ELO = 'elo'

