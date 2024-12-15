# Copyright (c) Alibaba, Inc. and its affiliates.
import os
from modelscope.utils.constant import DEFAULT_REPOSITORY_REVISION
from modelscope.utils.file_utils import get_dataset_cache_root, get_model_cache_root

DEFAULT_WORK_DIR = './outputs'
DEFAULT_MODEL_REVISION = DEFAULT_REPOSITORY_REVISION  # master
DEFAULT_MODEL_CACHE_DIR = get_model_cache_root()  # ~/.cache/modelscope/hub
DEFAULT_DATASET_CACHE_DIR = get_dataset_cache_root()  # ~/.cache/modelscope/datasets


class HubType:
    MODELSCOPE = 'modelscope'
    HUGGINGFACE = 'huggingface'
    LOCAL = 'local'


class DumpMode:
    OVERWRITE = 'overwrite'
    APPEND = 'append'


class MetricsConstant:
    EPSILON = float(1e-6)
    INVALID_VALUE = -9999999
    ROUGE_KEYS = [
        'rouge-1-r',
        'rouge-1-p',
        'rouge-1-f',
        'rouge-2-r',
        'rouge-2-p',
        'rouge-2-f',
        'rouge-l-r',
        'rouge-l-p',
        'rouge-l-f',
    ]


class MetricMembers:

    # Math accuracy metric
    MATH_ACCURACY = 'math_accuracy'

    # Code pass@k metric
    CODE_PASS_K = 'code_pass_k'

    # Code rouge metric
    ROUGE = 'rouge'

    # ELO rating system for pairwise comparison
    ELO = 'elo'

    # Pairwise comparison win/lose and tie(optional)
    PAIRWISE = 'pairwise'

    # Rating score for single model
    SCORE = 'score'


class ArenaWinner:

    MODEL_A = 'model_a'

    MODEL_B = 'model_b'

    TIE = 'tie'

    TIE_BOTH_BAD = 'tie_both_bad'

    UNKNOWN = 'unknown'


class ArenaMode:
    SINGLE = 'single'
    PAIRWISE = 'pairwise'
    PAIRWISE_BASELINE = 'pairwise_baseline'


class OutputsStructure:
    LOGS_DIR = 'logs'
    PREDICTIONS_DIR = 'predictions'
    REVIEWS_DIR = 'results'
    REPORTS_DIR = 'summary'
    CONFIGS_DIR = 'configs'

    def __init__(self, outputs_dir: str, is_make: bool = True):
        self.outputs_dir = outputs_dir
        self.logs_dir = os.path.join(outputs_dir, OutputsStructure.LOGS_DIR)
        self.predictions_dir = os.path.join(outputs_dir, OutputsStructure.PREDICTIONS_DIR)
        self.reviews_dir = os.path.join(outputs_dir, OutputsStructure.REVIEWS_DIR)
        self.reports_dir = os.path.join(outputs_dir, OutputsStructure.REPORTS_DIR)
        self.configs_dir = os.path.join(outputs_dir, OutputsStructure.CONFIGS_DIR)

        if is_make:
            self.create_directories()

    def create_directories(self):
        os.makedirs(self.outputs_dir, exist_ok=True)
        os.makedirs(self.logs_dir, exist_ok=True)
        os.makedirs(self.predictions_dir, exist_ok=True)
        os.makedirs(self.reviews_dir, exist_ok=True)
        os.makedirs(self.reports_dir, exist_ok=True)
        os.makedirs(self.configs_dir, exist_ok=True)


class AnswerKeys:
    ANSWER_ID = 'answer_id'
    RAW_INPUT = 'raw_input'
    ORIGIN_PROMPT = 'origin_prompt'
    MODEL_SPEC = 'model_spec'
    SUBSET_NAME = 'subset_name'
    CHOICES = 'choices'


class ReviewKeys:
    REVIEW_ID = 'review_id'
    REVIEWED = 'reviewed'
    REVIEWER_SPEC = 'reviewer_spec'
    REVIEW_TIME = 'review_time'
    MESSAGE = 'message'
    CONTENT = 'content'
    GOLD = 'gold'
    PRED = 'pred'
    RESULT = 'result'
    REVIEW = 'review'


class EvalConfigKeys:
    CLASS_REF = 'ref'
    CLASS_ARGS = 'args'
    ENABLE = 'enable'
    POSITION_BIAS_MITIGATION = 'position_bias_mitigation'
    RANDOM_SEED = 'random_seed'
    FN_COMPLETION_PARSER = 'fn_completion_parser'
    COMPLETION_PARSER_KWARGS = 'completion_parser_kwargs'
    OUTPUT_FILE = 'output_file'
    MODEL_ID_OR_PATH = 'model_id_or_path'
    MODEL_REVISION = 'revision'
    GENERATION_CONFIG = 'generation_config'
    PRECISION = 'precision'
    TEMPLATE_TYPE = 'template_type'


class FnCompletionParser:
    LMSYS_PARSER: str = 'lmsys_parser'
    RANKING_PARSER: str = 'ranking_parser'


class PositionBiasMitigation:
    NONE = 'none'
    RANDOMIZE_ORDER = 'randomize_order'
    SWAP_POSITION = 'swap_position'


class EvalStage:
    # Enums: `all`, `infer`, `review`
    ALL = 'all'
    INFER = 'infer'
    REVIEW = 'review'


class EvalType:

    CUSTOM = 'custom'
    CHECKPOINT = 'checkpoint'


class EvalBackend:
    # Use native evaluation pipeline of EvalScope
    NATIVE = 'Native'

    # Use OpenCompass framework as the evaluation backend
    OPEN_COMPASS = 'OpenCompass'

    # Use VLM Eval Kit as the multi-modal model evaluation backend
    VLM_EVAL_KIT = 'VLMEvalKit'

    # Use RAGEval as the RAG evaluation backend
    RAG_EVAL = 'RAGEval'

    # Use third-party evaluation backend/modules
    THIRD_PARTY = 'ThirdParty'
