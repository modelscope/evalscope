# Copyright (c) Alibaba, Inc. and its affiliates.
# flake8: noqa: E501
import copy
import json
import os
from argparse import Namespace
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union

from evalscope.api.model import GenerateConfig, Model, ModelAPI
from evalscope.constants import (
    DEFAULT_DATASET_CACHE_DIR,
    DEFAULT_WORK_DIR,
    EvalBackend,
    EvalType,
    HubType,
    JudgeStrategy,
    ModelTask,
)
from evalscope.utils.argument_utils import BaseArgument, parse_int_or_float
from evalscope.utils.deprecation_utils import deprecated_warning
from evalscope.utils.import_utils import check_import
from evalscope.utils.io_utils import dict_to_yaml, gen_hash, json_to_dict, safe_filename, yaml_to_dict
from evalscope.utils.logger import get_logger
from evalscope.version import __version__ as evalscope_version

logger = get_logger()

# Default configurations
DEFAULT_IMAGE_GEN_CONFIG = {
    'height': 1024,
    'width': 1024,
    'num_inference_steps': 50,
    'guidance_scale': 9.0,
}

DEFAULT_TEXT_GEN_CHECKPOINT_CONFIG = {
    'max_tokens': 2048,
    'do_sample': False,
    'top_k': 50,
    'top_p': 1.0,
    'temperature': 1.0,
    'n': 1,
}

DEFAULT_TEXT_GEN_SERVICE_CONFIG = {
    'temperature': 0.0,
}

DEFAULT_MODEL_ARGS_CHECKPOINT = {
    'revision': 'master',
    'precision': 'torch.float16',
}


@dataclass
class TaskConfig(BaseArgument):
    # Model-related arguments
    model: Optional[Union[str, Model, ModelAPI]] = None
    """The model to be evaluated. Can be a string path, Model object, or ModelAPI object."""

    model_id: Optional[str] = None
    """Unique identifier for the model. Auto-generated from model name if not provided."""

    model_args: Dict = field(default_factory=dict)
    """Additional arguments to pass to the model during initialization."""

    model_task: str = ModelTask.TEXT_GENERATION
    """The type of task the model performs (e.g., text generation, image generation)."""

    # Template-related arguments
    chat_template: Optional[str] = None
    """Chat template to use for formatting conversations with the model."""

    # Dataset-related arguments
    datasets: List[str] = field(default_factory=list)
    """List of dataset names to evaluate the model on."""

    dataset_args: Dict = field(default_factory=dict)
    """Additional arguments to pass to datasets during loading."""

    dataset_dir: str = DEFAULT_DATASET_CACHE_DIR
    """Directory where datasets are cached locally."""

    dataset_hub: str = HubType.MODELSCOPE
    """Hub platform to download datasets from (e.g., ModelScope, HuggingFace)."""

    repeats: int = 1
    """Number of times to repeat the dataset items for k-metrics evaluation."""

    # Generation configuration arguments
    generation_config: Union[Dict, GenerateConfig] = field(default_factory=dict)
    """Configuration parameters for text/image generation."""

    # Evaluation-related arguments
    eval_type: Optional[str] = None
    """Type of evaluation: checkpoint, service, or mock."""

    eval_backend: str = EvalBackend.NATIVE
    """Backend framework to use for evaluation."""

    eval_config: Union[str, Dict, None] = None
    """Additional evaluation configuration parameters."""

    limit: Optional[Union[int, float]] = None
    """Maximum number of samples to evaluate. Can be int (count) or float (fraction)."""

    eval_batch_size: int = 1
    """Batch size for evaluation processing."""

    # Cache and working directory arguments
    use_cache: Optional[str] = None
    """Whether to use cached results and which cache strategy to apply."""

    rerun_review: bool = False
    """Whether to rerun the review process even if results exist."""

    work_dir: str = DEFAULT_WORK_DIR
    """Working directory for storing evaluation results and temporary files."""

    no_timestamp: bool = False
    """Do not add timestamp to the work_dir to avoid overwriting previous results."""

    # Debug and runtime mode arguments
    ignore_errors: bool = False
    """Whether to continue evaluation when encountering errors."""

    debug: bool = False
    """Enable debug mode for detailed logging and error reporting."""

    seed: Optional[int] = 42
    """Random seed for reproducible results."""

    api_url: Optional[str] = None
    """API endpoint URL for server-based model evaluation."""

    api_key: Optional[str] = 'EMPTY'
    """API key for authenticating with server-based models."""

    timeout: Optional[float] = None
    """Request timeout in seconds for server-based models."""

    stream: Optional[bool] = None
    """Whether to use streaming responses for server-based models."""

    # LLMJudge arguments
    judge_strategy: str = JudgeStrategy.AUTO
    """Strategy for LLM-based judgment (auto, single, pairwise)."""

    judge_worker_num: int = 1
    """Number of worker processes for parallel LLM judging."""

    judge_model_args: Optional[Dict] = field(default_factory=dict)
    """Additional arguments for the judge model configuration."""

    analysis_report: bool = False
    """Whether to generate detailed analysis reports after evaluation."""

    # Sandbox configuration arguments
    use_sandbox: bool = False
    """Whether to execute code in a sandboxed environment."""

    sandbox_type: Optional[str] = 'docker'
    """Type of sandbox environment for code execution (e.g., docker). Default is 'docker'."""

    sandbox_manager_config: Optional[Dict] = field(default_factory=dict)
    """Configuration for the sandbox manager. Default is local manager. If url is provided, it will use remote manager."""

    evalscope_version: Optional[str] = evalscope_version
    """EvalScope version used for the evaluation."""

    def __post_init__(self):
        self.__init_model_and_id()

        self.__init_eval_data_config()
        self.__init_eval_config()

        # Set default generation_config and model_args
        self.__init_default_generation_config()
        self.__init_default_model_args()
        self.__init_default_sandbox_config()

    def __init_model_and_id(self):
        # Set model to DummyCustomModel if not provided
        if self.model is None:
            logger.info('No model is provided, using DummyCustomModel for testing.')
            self.model = self.model_task
            self.eval_type = EvalType.MOCK_LLM

        # Set eval_type to SERVICE if api_url is provided
        if self.api_url is not None and self.eval_type is None:
            logger.info('api_url is provided, setting eval_type to SERVICE.')
            self.eval_type = EvalType.SERVICE

        # Set eval_type to CHECKPOINT if model is a string path and eval_type is not set
        if self.model and self.eval_type is None:
            logger.info('No eval_type is provided, setting eval_type to CHECKPOINT.')
            self.eval_type = EvalType.CHECKPOINT

        # Set model_id if not provided
        if not self.model_id:
            self.model_id = self._infer_model_id()

    def _infer_model_id(self) -> str:
        if isinstance(self.model, str):
            return safe_filename(os.path.basename(self.model))
        elif isinstance(self.model, Model):
            return safe_filename(self.model.name)
        elif isinstance(self.model, ModelAPI):
            return safe_filename(self.model.model_name)
        return 'dummy_model'

    def __init_eval_data_config(self):
        # Post process limit
        if self.limit is not None:
            self.limit = parse_int_or_float(self.limit)

    def __init_default_generation_config(self):
        # 1. Set defaults if empty
        if not self.generation_config:
            self.generation_config = self._get_default_generation_config()

        # 2. Validate/Convert to GenerateConfig object
        if isinstance(self.generation_config, dict):
            self.generation_config = GenerateConfig.model_validate(self.generation_config)

        # 3. Sync batch size
        self.generation_config.batch_size = self.eval_batch_size

        # 4. Handle deprecations
        self._handle_generation_config_deprecations()

    def _get_default_generation_config(self) -> Dict:
        if self.model_task == ModelTask.IMAGE_GENERATION:
            if self.eval_batch_size != 1:
                logger.warning('For image generation task, we only support eval_batch_size=1 for now, changed to 1.')
                self.eval_batch_size = 1
            return DEFAULT_IMAGE_GEN_CONFIG.copy()

        elif self.model_task == ModelTask.TEXT_GENERATION:
            if self.eval_type == EvalType.CHECKPOINT:
                return DEFAULT_TEXT_GEN_CHECKPOINT_CONFIG.copy()
            elif self.eval_type == EvalType.SERVICE:
                return DEFAULT_TEXT_GEN_SERVICE_CONFIG.copy()

        return {}

    def _handle_generation_config_deprecations(self):
        if self.timeout is not None:
            deprecated_warning(
                logger,
                'The `timeout` parameter is deprecated and will be removed in v2.0.0. Use `generation_config.timeout` instead.'
            )
            self.generation_config.timeout = self.timeout

        if self.stream is not None:
            deprecated_warning(
                logger,
                'The `stream` parameter is deprecated and will be removed in v2.0.0. Use `generation_config.stream` instead.'
            )
            self.generation_config.stream = self.stream

        if self.generation_config.n is not None and self.generation_config.n > 1:
            self.repeats = self.generation_config.n
            self.generation_config.n = 1
            deprecated_warning(
                logger,
                'The `n` parameter in generation_config is deprecated and will be removed in v2.0.0. Use `TaskConfig.repeats` instead.'
            )

    def __init_default_model_args(self):
        if self.model_args:
            return
        if self.model_task == ModelTask.TEXT_GENERATION and self.eval_type == EvalType.CHECKPOINT:
            self.model_args = DEFAULT_MODEL_ARGS_CHECKPOINT.copy()

    def __init_default_sandbox_config(self):
        if not self.use_sandbox:
            return
        check_import('ms_enclave', 'evalscope[sandbox]', raise_error=True)

        if not self.sandbox_type:
            self.sandbox_type = 'docker'

    def __init_eval_config(self):
        if not self.eval_config:
            return
        if isinstance(self.eval_config, dict):
            return
        if isinstance(self.eval_config, str):
            extension = os.path.splitext(self.eval_config)[-1]
            if extension in ['.yaml', '.yml']:
                self.eval_config = yaml_to_dict(self.eval_config)
            elif extension == '.json':
                self.eval_config = json_to_dict(self.eval_config)
            else:
                try:
                    self.eval_config = json.loads(self.eval_config)
                except Exception as e:
                    raise ValueError('eval_config string is not a valid json string or file path.') from e
        else:
            raise ValueError('eval_config should be a dict or a file path string.')

    def update(self, other: Union['TaskConfig', dict]):
        if isinstance(other, TaskConfig):
            other = other.to_dict()
        self.__dict__.update(other)

    def dump_yaml(self, output_dir: str):
        """Dump the task configuration to a YAML file."""
        task_cfg_file = os.path.join(output_dir, f'task_config_{gen_hash(str(self), bits=6)}.yaml')
        try:
            logger.info(f'Dump task config to {task_cfg_file}')
            dict_to_yaml(self.to_dict(), task_cfg_file)
        except Exception as e:
            logger.warning(f'Failed to dump overall task config: {e}')

    def to_dict(self):
        result = copy.copy(self.__dict__)

        # Remove sensitive info
        result.pop('api_key', None)

        # Handle nested sensitive info in judge_model_args
        if self.judge_model_args:
            result['judge_model_args'] = copy.deepcopy(self.judge_model_args)
            result['judge_model_args'].pop('api_key', None)

        # Serialize Model objects
        if isinstance(self.model, (Model, ModelAPI)):
            result['model'] = self.model.__class__.__name__

        # Serialize GenerateConfig
        if isinstance(self.generation_config, GenerateConfig):
            result['generation_config'] = self.generation_config.model_dump(exclude_unset=True)

        return result


def parse_task_config(task_cfg) -> TaskConfig:
    """Parse task configuration from various formats into a TaskConfig object."""
    if isinstance(task_cfg, TaskConfig):
        logger.info('Args: Task config is provided with TaskConfig type.')
    elif isinstance(task_cfg, dict):
        logger.info('Args: Task config is provided with dictionary type.')
        task_cfg = TaskConfig.from_dict(task_cfg)
    elif isinstance(task_cfg, Namespace):
        logger.info('Args: Task config is provided with CommandLine type.')
        task_cfg = TaskConfig.from_args(task_cfg)
    elif isinstance(task_cfg, str):
        extension = os.path.splitext(task_cfg)[-1]
        logger.info(f'Args: Task config is provided with {extension} file type.')
        if extension in ['.yaml', '.yml']:
            task_cfg = TaskConfig.from_yaml(task_cfg)
        elif extension == '.json':
            task_cfg = TaskConfig.from_json(task_cfg)
        else:
            raise ValueError('Args: Unsupported file extension.')
    else:
        raise ValueError('Args: Please provide a valid task config.')
    return task_cfg
