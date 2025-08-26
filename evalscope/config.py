# Copyright (c) Alibaba, Inc. and its affiliates.
# flake8: noqa: E501
import copy
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
    OutputType,
)
from evalscope.utils.argument_utils import BaseArgument, parse_int_or_float
from evalscope.utils.deprecation_utils import deprecated_warning
from evalscope.utils.io_utils import dict_to_yaml, gen_hash, safe_filename
from evalscope.utils.logger import get_logger

logger = get_logger()


@dataclass
class TaskConfig(BaseArgument):
    # Model-related arguments
    model: Optional[Union[str, Model, ModelAPI]] = None
    model_id: Optional[str] = None
    model_args: Dict = field(default_factory=dict)
    model_task: str = ModelTask.TEXT_GENERATION

    # Template-related arguments
    chat_template: Optional[str] = None

    # Dataset-related arguments
    datasets: List[str] = field(default_factory=list)
    dataset_args: Dict = field(default_factory=dict)
    dataset_dir: str = DEFAULT_DATASET_CACHE_DIR
    dataset_hub: str = HubType.MODELSCOPE
    repeats: int = 1  # Number of times to repeat the dataset items for k-metrics

    # Generation configuration arguments
    generation_config: Union[Dict, GenerateConfig] = field(default_factory=dict)

    # Evaluation-related arguments
    eval_type: str = EvalType.CHECKPOINT
    eval_backend: str = EvalBackend.NATIVE
    eval_config: Union[str, Dict, None] = None
    limit: Optional[Union[int, float]] = None
    eval_batch_size: int = 1

    # Cache and working directory arguments
    use_cache: Optional[str] = None
    rerun_review: bool = False
    work_dir: str = DEFAULT_WORK_DIR

    # Debug and runtime mode arguments
    ignore_errors: bool = False
    debug: bool = False
    seed: Optional[int] = 42
    api_url: Optional[str] = None  # Only used for server model
    api_key: Optional[str] = 'EMPTY'  # Only used for server model
    timeout: Optional[float] = None  # Only used for server model
    stream: Optional[bool] = None  # Only used for server model

    # LLMJudge arguments
    judge_strategy: str = JudgeStrategy.AUTO
    judge_worker_num: int = 1
    judge_model_args: Optional[Dict] = field(default_factory=dict)
    analysis_report: bool = False

    def __post_init__(self):
        self.__init_model_and_id()

        self.__init_eval_data_config()

        # Set default generation_config and model_args
        self.__init_default_generation_config()
        self.__init_default_model_args()

    def __init_model_and_id(self):
        # Set model to DummyCustomModel if not provided
        if self.model is None:
            self.model = self.model_task
            self.eval_type = EvalType.MOCK_LLM
        else:
            if self.model_task == ModelTask.IMAGE_GENERATION:
                self.eval_type = EvalType.TEXT2IMAGE

        # Set model_id if not provided
        if not self.model_id:
            if isinstance(self.model, str):
                self.model_id = safe_filename(os.path.basename(self.model))
            elif isinstance(self.model, Model):
                self.model_id = safe_filename(self.model.name)
            elif isinstance(self.model, ModelAPI):
                self.model_id = safe_filename(self.model.model_name)
            else:
                self.model_id = 'dummy_model'

    def __init_eval_data_config(self):
        # Post process limit
        if self.limit is not None:
            self.limit = parse_int_or_float(self.limit)

    def __init_default_generation_config(self):
        if not self.generation_config:
            if self.model_task == ModelTask.IMAGE_GENERATION:
                self.generation_config = {
                    'height': 1024,
                    'width': 1024,
                    'num_inference_steps': 50,
                    'guidance_scale': 9.0,
                }
            elif self.model_task == ModelTask.TEXT_GENERATION:
                if self.eval_type == EvalType.CHECKPOINT:
                    self.generation_config = {
                        'max_tokens': 2048,
                        'do_sample': False,
                        'top_k': 50,
                        'top_p': 1.0,
                        'temperature': 1.0,
                        'n': 1,
                    }
                elif self.eval_type == EvalType.SERVICE:
                    self.generation_config = {
                        'max_tokens': 2048,
                        'temperature': 0.0,
                    }
        if isinstance(self.generation_config, dict):
            self.generation_config = GenerateConfig.model_validate(self.generation_config)

        # Set eval_batch_size to generation_config.batch_size
        self.generation_config.batch_size = self.eval_batch_size

        # Set default values for generation_config
        if self.timeout is not None:
            deprecated_warning(
                logger,
                'The `timeout` parameter is deprecated and will be removed in v1.1.0. Use `generation_config.timeout` instead.'
            )
            self.generation_config.timeout = self.timeout

        if self.stream is not None:
            deprecated_warning(
                logger,
                'The `stream` parameter is deprecated and will be removed in v1.1.0. Use `generation_config.stream` instead.'
            )
            self.generation_config.stream = self.stream

        if self.generation_config.n is not None and self.generation_config.n > 1:
            self.repeats = self.generation_config.n
            self.generation_config.n = 1
            deprecated_warning(
                logger,
                'The `n` parameter in generation_config is deprecated and will be removed in v1.1.0. Use `TaskConfig.repeats` instead.'
            )

    def __init_default_model_args(self):
        if self.model_args:
            return
        if self.model_task == ModelTask.TEXT_GENERATION:
            if self.eval_type == EvalType.CHECKPOINT:
                self.model_args = {
                    'revision': 'master',
                    'precision': 'torch.float16',
                }

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
        result = copy.deepcopy(self.__dict__)
        del result['api_key']  # Do not expose api_key in the config

        if isinstance(self.model, (Model, ModelAPI)):
            result['model'] = self.model.__class__.__name__

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
