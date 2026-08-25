# Copyright (c) Alibaba, Inc. and its affiliates.
# flake8: noqa
import os
from datetime import timedelta, timezone
from enum import Enum

os.environ['MODELSCOPE_LOG_LEVEL'] = '40'  # Set default log level to ERROR


def _get_modelscope_cache_dir() -> str:
    return os.path.expanduser(os.getenv('MODELSCOPE_CACHE', '~/.cache/modelscope/hub'))


DEFAULT_WORK_DIR = './outputs'
DEFAULT_MODEL_REVISION = 'master'
DEFAULT_MODEL_CACHE_DIR = os.path.join(_get_modelscope_cache_dir(), 'models')  # ~/.cache/modelscope/hub/models
DEFAULT_DATASET_CACHE_DIR = os.path.join(_get_modelscope_cache_dir(), 'datasets')  # ~/.cache/modelscope/hub/datasets
DEFAULT_ROOT_CACHE_DIR = DEFAULT_DATASET_CACHE_DIR  # compatible with old version
DEFAULT_EVALSCOPE_CACHE_DIR = os.path.expanduser(
    os.getenv('EVALSCOPE_CACHE', '~/.cache/evalscope')
)  # ~/.cache/evalscope
DATASET_TRANSFORM_BATCH_SIZE = int(os.getenv('DATASET_TF_BATCH_SIZE', '100'))
HEARTBEAT_INTERVAL_SEC = int(os.getenv('EVALSCOPE_HEARTBEAT_INTERVAL', '60'))  # 60 seconds
DEFAULT_LANGUAGE = os.getenv('EVALSCOPE_LANGUAGE', 'en')  # default language: 'en' or 'zh'
USE_OSS = os.getenv('USE_OSS', '0') == '1'  # whether to use OSS/FUSE-mounted filesystem
BEIJING_TZ = timezone(timedelta(hours=8))  # UTC+8


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
    TEXT2IMAGE = 'text2image'  # image generation service
    TEXT2SPEECH = 'text2speech'  # text-to-speech service
    IMAGE_EDITING = 'image_editing'  # image editing service
    OPENAI_API = 'openai_api'
    OPENAI_RESPONSES_API = 'openai_responses_api'
    ANTHROPIC_API = 'anthropic_api'
    LITELLM = 'litellm'


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


class ScoringPolicy(str, Enum):
    """What a benchmark's own scoring paths can do, declared by its adapter.

    Encodes two orthogonal facts in one value so that the illegal combination -- no usable rule
    path yet ``auto`` resolving to rule -- cannot be expressed. Every benchmark always has a
    judge path because ``DataAdapter`` inherits the generic grader from ``LLMJudgeMixin``, so
    "supports a judge" needs no field.
    """

    RULE_DEFAULT = 'rule_default'
    """Usable rule path; ``auto`` scores by rule."""

    JUDGE_DEFAULT = 'judge_default'
    """Usable rule path, but ``auto`` scores with the judge because it is more faithful."""

    JUDGE_ONLY = 'judge_only'
    """No usable rule path: rule scoring would raise or only ever emit zeros."""

    @property
    def rule_supported(self) -> bool:
        return self is not ScoringPolicy.JUDGE_ONLY

    @property
    def judge_by_default(self) -> bool:
        return self is not ScoringPolicy.RULE_DEFAULT


class JudgeScoreType:
    NUMERIC = 'numeric'  # numeric score
    PATTERN = 'pattern'  # pattern matching score


class ScoreStatus(str, Enum):
    """Whether a score is usable, and why it is not.

    A failed judge is not a zero: only ``SUCCESS`` and ``FALLBACK`` carry a score that may
    enter aggregation. The remaining values mean the score is unavailable and the sample
    must be excluded from the affected metric rather than counted as 0.
    """

    SUCCESS = 'success'
    TRANSPORT_ERROR = 'transport_error'
    PARSE_ERROR = 'parse_error'
    INVALID_SESSION = 'invalid_session'
    FALLBACK = 'fallback'
    DEGRADED = 'degraded'
    EXCLUDED = 'excluded'

    @property
    def is_usable(self) -> bool:
        return self in (ScoreStatus.SUCCESS, ScoreStatus.FALLBACK, ScoreStatus.DEGRADED)


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
    TEXT_TO_SPEECH = 'TextToSpeech'
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
    IMAGE_CAPTIONING = 'ImageCaptioning'
    VIDEO = 'Video'


class FileConstants:
    IMAGE_PATH = 'image_path'
    ID = 'id'


class VisualizerType:
    WANDB = 'wandb'
    SWANLAB = 'swanlab'
    CLEARML = 'clearml'


# --- Report / Visualization constants (migrated from app.constants) ---
PLOTLY_THEME = 'plotly_dark'
PLOTLY_CDN_URL = 'https://resources.modelscope.cn/third-part/js/plotly/plotly-2.35.2.min.js'
DEFAULT_BAR_WIDTH = 0.2
LATEX_DELIMITERS = [{
    'left': '$$',
    'right': '$$',
    'display': True,
}, {
    'left': '$',
    'right': '$',
    'display': False,
}, {
    'left': '\\(',
    'right': '\\)',
    'display': False,
}, {
    'left': '\\[',
    'right': '\\]',
    'display': True,
}]


class LoggingConstants:
    DATE_FORMAT = '%Y-%m-%d %H:%M:%S'
    # Console output formats (colorlog)
    COLOR_DETAILED_FORMAT = (
        '%(asctime)s - %(name)s - %(filename)s - %(funcName)s - %(lineno)d'
        ' - %(log_color)s%(levelname)s%(reset)s: %(message)s'
    )
    COLOR_SIMPLE_FORMAT = ('%(asctime)s - %(name)s - %(log_color)s%(levelname)s%(reset)s: %(message)s')
    # File output formats (plain)
    DETAILED_FORMAT = (
        '%(asctime)s - %(name)s - %(filename)s - %(funcName)s - %(lineno)d'
        ' - %(levelname)s: %(message)s'
    )
    SIMPLE_FORMAT = '%(asctime)s - %(name)s - %(levelname)s: %(message)s'
