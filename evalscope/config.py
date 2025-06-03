# Copyright (c) Alibaba, Inc. and its affiliates.

import copy
import json
import os
from argparse import Namespace
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union

from evalscope.constants import (DEFAULT_DATASET_CACHE_DIR, DEFAULT_WORK_DIR, EvalBackend, EvalStage, EvalType, HubType,
                                 JudgeStrategy, ModelTask, OutputType)
from evalscope.models import CustomModel, DummyCustomModel
from evalscope.utils import gen_hash
from evalscope.utils.io_utils import dict_to_yaml, json_to_dict, yaml_to_dict
from evalscope.utils.logger import get_logger
from evalscope.utils.utils import parse_int_or_float

logger = get_logger()

cur_path = os.path.dirname(os.path.abspath(__file__))


@dataclass
class TaskConfig:
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
        if self.model is None:
            self.model = DummyCustomModel()
            self.eval_type = EvalType.CUSTOM

        if (not self.model_id) and self.model:
            if isinstance(self.model, CustomModel):
                self.model_id = self.model.config.get('model_id', 'custom_model')
            else:
                self.model_id = os.path.basename(self.model).rstrip(os.sep)
            # fix path error, see http://github.com/modelscope/evalscope/issues/377
            self.model_id = self.model_id.replace(':', '-')

        # Set default eval_batch_size based on eval_type
        if self.eval_batch_size is None:
            self.eval_batch_size = 8 if self.eval_type == EvalType.SERVICE else 1

        # Post process limit
        if self.limit is not None:
            self.limit = parse_int_or_float(self.limit)

        # Set default generation_config and model_args
        self.__init_default_generation_config()
        self.__init_default_model_args()

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

    def to_dict(self):
        result = self.__dict__.copy()
        if isinstance(self.model, CustomModel):
            result['model'] = self.model.__class__.__name__
        return result

    def __str__(self):
        return json.dumps(self.to_dict(), indent=4, default=str, ensure_ascii=False)

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

    @staticmethod
    def list():
        return list(registry_tasks.keys())

    @staticmethod
    def from_yaml(yaml_file: str):
        return TaskConfig.from_dict(yaml_to_dict(yaml_file))

    @staticmethod
    def from_dict(d: dict):
        return TaskConfig(**d)

    @staticmethod
    def from_json(json_file: str):
        return TaskConfig.from_dict(json_to_dict(json_file))

    @staticmethod
    def from_args(args: Namespace):
        # Convert Namespace to a dictionary and filter out None values
        args_dict = {k: v for k, v in vars(args).items() if v is not None}

        if 'func' in args_dict:
            del args_dict['func']  # Note: compat CLI arguments

        return TaskConfig.from_dict(args_dict)

    @staticmethod
    def load(custom_model: CustomModel, tasks: List[str]) -> List['TaskConfig']:
        res_list = []
        for task_name in tasks:
            task = registry_tasks.get(task_name, None)
            if task is None:
                logger.error(f'No task found in tasks: {list(registry_tasks.keys())}, got task_name: {task_name}')
                continue

            task.model = custom_model
            task.model_args = custom_model.config
            task.model_id = type(custom_model).__name__
            res_list.append(task)

        return res_list

    @staticmethod
    def registry(name: str, data_pattern: str, dataset_dir: str = None, subset_list: list = None) -> None:
        """
        Register a new task (dataset) for evaluation.

        Args:
            name: str, the dataset name.
            data_pattern: str, the data pattern for the task.
                    e.g. `mmlu`, `ceval`, `gsm8k`, ...
                    refer to task_config.list() for all available datasets.
            dataset_dir: str, the directory to store multiple datasets files. e.g. /path/to/data,
                then your specific custom dataset directory will be /path/to/data/{name}
            subset_list: list, the subset list for the dataset.
                e.g. ['middle_school_politics', 'operating_system']
                refer to the mmlu for example.  https://github.com/hendrycks/test/blob/master/categories.py
        """
        available_datasets = list(registry_tasks.keys())
        if data_pattern not in available_datasets:
            logger.error(
                f'No dataset found in available datasets: {available_datasets}, got data_pattern: {data_pattern}')
            return

        # Reuse the existing task config and update the datasets
        pattern_config = registry_tasks[data_pattern]

        custom_config = copy.deepcopy(pattern_config)
        custom_config.datasets = [data_pattern]
        custom_config.dataset_args = {data_pattern: {}}
        custom_config.eval_type = EvalType.CHECKPOINT

        if dataset_dir is not None:
            custom_config.dataset_args[data_pattern].update({'local_path': dataset_dir})

        if subset_list is not None:
            custom_config.dataset_args[data_pattern].update({'subset_list': subset_list})

        registry_tasks.update({name: custom_config})
        logger.info(f'** Registered task: {name} with data pattern: {data_pattern}')


tasks = ['arc', 'gsm8k', 'mmlu', 'cmmlu', 'ceval', 'bbh', 'general_qa']

registry_tasks = {task: TaskConfig.from_yaml(os.path.join(cur_path, f'registry/tasks/{task}.yaml')) for task in tasks}


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


class TempModel(CustomModel):

    def __init__(self, config: dict):
        super().__init__(config=config)

    def predict(self, prompts: str, **kwargs):
        return [item + ': response' for item in prompts]


if __name__ == '__main__':
    model = TempModel(config={'model_id': 'test-swift-dummy-model'})
    task_config = TaskConfig()

    # Register a new task
    TaskConfig.registry(name='arc_swift', data_pattern='arc', dataset_dir='/path/to/swift_custom_work')

    swift_eval_task: List[TaskConfig] = TaskConfig.load(custom_model=model, tasks=['gsm8k', 'arc', 'arc_swift'])
    for item in swift_eval_task:
        print(item)
        print()
