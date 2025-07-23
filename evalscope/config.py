# Copyright (c) Alibaba, Inc. and its affiliates.

import copy
import os
from argparse import Namespace
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union

from evalscope.constants import (DEFAULT_DATASET_CACHE_DIR, DEFAULT_WORK_DIR, EvalBackend, EvalStage, EvalType, HubType,
                                 JudgeStrategy, ModelTask, OutputType)
from evalscope.models import CustomModel, DummyCustomModel, DummyT2IModel
from evalscope.utils.argument_utils import BaseArgument, parse_int_or_float
from evalscope.utils.io_utils import dict_to_yaml, gen_hash
from evalscope.utils.logger import get_logger

logger = get_logger()


@dataclass
class TaskConfig(BaseArgument):
    # Model-related arguments
    model: Union[str, 'CustomModel', None] = None
    model_id: Optional[str] = None
    model_args: Dict = field(default_factory=dict)
    model_task: str = ModelTask.TEXT_GENERATION

    # Template-related arguments
    template_type: Optional[str] = None  # Deprecated, will be removed in v1.0.0.
    chat_template: Optional[str] = None

    # Dataset-related arguments
    datasets: List[str] = field(default_factory=list)
    dataset_args: Dict = field(default_factory=dict)
    dataset_dir: str = DEFAULT_DATASET_CACHE_DIR
    dataset_hub: str = HubType.MODELSCOPE

    # Generation configuration arguments
    generation_config: Dict = field(default_factory=dict)

    # Evaluation-related arguments
    eval_type: str = EvalType.CHECKPOINT
    eval_backend: str = EvalBackend.NATIVE
    eval_config: Union[str, Dict, None] = None
    stage: str = EvalStage.ALL
    limit: Optional[Union[int, float]] = None
    eval_batch_size: Optional[int] = None

    # Cache and working directory arguments
    mem_cache: bool = False  # Deprecated, will be removed in v1.0.0.
    use_cache: Optional[str] = None
    work_dir: str = DEFAULT_WORK_DIR
    outputs: Optional[str] = None  # Deprecated, will be removed in v1.0.0.

    # Debug and runtime mode arguments
    ignore_errors: bool = False
    debug: bool = False
    dry_run: bool = False
    seed: Optional[int] = 42
    api_url: Optional[str] = None  # Only used for server model
    api_key: Optional[str] = 'EMPTY'  # Only used for server model
    timeout: Optional[float] = None  # Only used for server model
    stream: bool = False  # Only used for server model

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
            self.eval_type = EvalType.CUSTOM
            if self.model_task == ModelTask.IMAGE_GENERATION:
                self.model = DummyT2IModel()
            else:
                self.model = DummyCustomModel()

        # Set model_id if not provided
        if (not self.model_id) and self.model:
            if isinstance(self.model, CustomModel):
                self.model_id = self.model.config.get('model_id', 'custom_model')
            else:
                self.model_id = os.path.basename(self.model).rstrip(os.sep)
            # fix path error, see http://github.com/modelscope/evalscope/issues/377
            self.model_id = self.model_id.replace(':', '-').replace('/', '-')

    def __init_eval_data_config(self):
        # Set default eval_batch_size based on eval_type
        if self.eval_batch_size is None:
            self.eval_batch_size = 8 if self.eval_type == EvalType.SERVICE else 1

        # Post process limit
        if self.limit is not None:
            self.limit = parse_int_or_float(self.limit)

    def __init_default_generation_config(self):
        if self.generation_config:
            return
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
                    'max_length': 2048,
                    'max_new_tokens': 512,
                    'do_sample': False,
                    'top_k': 50,
                    'top_p': 1.0,
                    'temperature': 1.0,
                }
            elif self.eval_type == EvalType.SERVICE:
                self.generation_config = {
                    'max_tokens': 2048,
                    'temperature': 0.0,
                }

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
        result = self.__dict__.copy()
        if isinstance(self.model, CustomModel):
            result['model'] = self.model.__class__.__name__
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
