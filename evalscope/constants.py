# Copyright (c) Alibaba, Inc. and its affiliates.
# flake8: noqa
import os

os.environ['MODELSCOPE_LOG_LEVEL'] = '40'  # Set default log level to ERROR

from modelscope.utils.constant import DEFAULT_REPOSITORY_REVISION
from modelscope.utils.file_utils import get_dataset_cache_root, get_model_cache_root

DEFAULT_WORK_DIR = './outputs'
DEFAULT_MODEL_REVISION = DEFAULT_REPOSITORY_REVISION  # master
DEFAULT_MODEL_CACHE_DIR = get_model_cache_root()  # ~/.cache/modelscope/hub/models
DEFAULT_DATASET_CACHE_DIR = get_dataset_cache_root()  # ~/.cache/modelscope/hub/datasets
DEFAULT_ROOT_CACHE_DIR = DEFAULT_DATASET_CACHE_DIR  # compatible with old version
DEFAULT_EVALSCOPE_CACHE_DIR = os.path.expanduser(
    os.getenv('EVALSCOPE_CACHE', '~/.cache/evalscope')
)  # ~/.cache/evalscope
IS_BUILD_DOC = os.getenv('BUILD_DOC', '0') == '1'  # To avoid some heavy dependencies when building doc
HEARTBEAT_INTERVAL_SEC = 60  # 60 seconds


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


class ArenaWinner:

    MODEL_A = 'model_a'
    MODEL_B = 'model_b'
    TIE = 'tie'
    TIE_BOTH_BAD = 'tie_both_bad'
    UNKNOWN = 'unknown'


class AnswerKeys:
    INDEX = 'index'
    ANSWER_ID = 'answer_id'
    RAW_INPUT = 'raw_input'
    ORIGIN_PROMPT = 'origin_prompt'
    MODEL_SPEC = 'model_spec'
    SUBSET_NAME = 'subset_name'
    CHOICES = 'choices'


class EvalType:

    CUSTOM = 'custom'
    MOCK_LLM = 'mock_llm'
    CHECKPOINT = 'llm_ckpt'  # native model checkpoint
    SERVICE = 'openai_api'  # model service
    TEXT2IMAGE = 'text2image'  # image generation service
    IMAGE_EDITING = 'image_editing'  # image editing service


class OutputType:
    LOGITS = 'logits'  # for logits output tasks
    GENERATION = 'generation'  # for text generation tasks and general tasks
    MULTIPLE_CHOICE = 'multiple_choice_logits'  # for multiple choice tasks
    CONTINUOUS = 'continuous_logits'  # for continuous tasks
    IMAGE_GENERATION = 'image_generation'  # for image generation tasks


class EvalBackend:
    NATIVE = 'Native'
    OPEN_COMPASS = 'OpenCompass'
    VLM_EVAL_KIT = 'VLMEvalKit'
    RAG_EVAL = 'RAGEval'
    THIRD_PARTY = 'ThirdParty'


class DataCollection:
    NAME = 'data_collection'
    INFO = 'collection_info'
    REPORT_NAME = 'collection_detailed_report.json'


class JudgeStrategy:
    AUTO = 'auto'
    RULE = 'rule'
    LLM = 'llm'
    LLM_RECALL = 'llm_recall'


class JudgeScoreType:
    NUMERIC = 'numeric'  # numeric score
    PATTERN = 'pattern'  # pattern matching score


class ModelTask:
    TEXT_GENERATION = 'text_generation'
    IMAGE_GENERATION = 'image_generation'


class Tags:
    KNOWLEDGE = 'Knowledge'
    MULTIPLE_CHOICE = 'MCQ'
    MATH = 'Math'
    REASONING = 'Reasoning'
    CODING = 'Coding'
    CHINESE = 'Chinese'
    COMMONSENSE = 'Commonsense'
    QA = 'QA'
    NER = 'NER'
    READING_COMPREHENSION = 'ReadingComprehension'
    CUSTOM = 'Custom'
    INSTRUCTION_FOLLOWING = 'InstructionFollowing'
    ARENA = 'Arena'
    LONG_CONTEXT = 'LongContext'
    RETRIEVAL = 'Retrieval'
    FUNCTION_CALLING = 'FunctionCalling'
    TEXT_TO_IMAGE = 'TextToImage'
    IMAGE_EDITING = 'ImageEditing'
    MULTI_MODAL = 'MultiModal'
    MULTI_LINGUAL = 'MultiLingual'
    MULTI_TURN = 'MultiTurn'
    YES_NO = 'Yes/No'
    HALLUCINATION = 'Hallucination'
    MEDICAL = 'Medical'
    AGENT = 'Agent'
    MT = 'MachineTranslation'
    GROUNDING = 'Grounding'
    SPEECH_RECOGNITION = 'SpeechRecognition'
    AUDIO = 'Audio'


class FileConstants:
    IMAGE_PATH = 'image_path'
    ID = 'id'


class VisualizerType:
    WANDB = 'wandb'
    SWANLAB = 'swanlab'
    CLEARML = 'clearml'
