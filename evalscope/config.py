# Copyright (c) Alibaba, Inc. and its affiliates.

import copy
import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union

from evalscope.constants import DEFAULT_DATASET_CACHE_DIR, DEFAULT_WORK_DIR, EvalBackend, EvalStage, EvalType, HubType
from evalscope.models.custom import CustomModel
from evalscope.utils import yaml_to_dict
from evalscope.utils.logger import get_logger

logger = get_logger()

cur_path = os.path.dirname(os.path.abspath(__file__))

registry_tasks = {
    'arc': yaml_to_dict(os.path.join(cur_path, 'registry/tasks/arc.yaml')),
    'gsm8k': yaml_to_dict(os.path.join(cur_path, 'registry/tasks/gsm8k.yaml')),
    'mmlu': yaml_to_dict(os.path.join(cur_path, 'registry/tasks/mmlu.yaml')),
    'cmmlu': yaml_to_dict(os.path.join(cur_path, 'registry/tasks/cmmlu.yaml')),
    'ceval': yaml_to_dict(os.path.join(cur_path, 'registry/tasks/ceval.yaml')),
    'bbh': yaml_to_dict(os.path.join(cur_path, 'registry/tasks/bbh.yaml')),
    'general_qa': yaml_to_dict(os.path.join(cur_path, 'registry/tasks/general_qa.yaml')),
}

DEFAULT_MODEL_ARGS = {'revision': 'master', 'precision': 'auto', 'device': 'auto'}
DEFAULT_GENERATION_CONFIG = {
    'max_length': 2048,
    'max_new_tokens': 512,
    'do_sample': False,
    'top_k': 50,
    'top_p': 1.0,
    'temperature': 1.0,
}


@dataclass
class TaskConfig:
    # Model-related arguments
    model: Union[str, CustomModel, None] = None
    model_args: Optional[Dict] = field(default_factory=lambda: DEFAULT_MODEL_ARGS | {})

    # Template-related arguments
    template_type: Optional[str] = None  # Deprecated, will be removed in v1.0.0.
    chat_template: Optional[str] = None

    # Dataset-related arguments
    datasets: Optional[List[str]] = None
    dataset_args: Optional[Dict] = field(default_factory=dict)
    dataset_dir: str = DEFAULT_DATASET_CACHE_DIR
    dataset_hub: str = HubType.MODELSCOPE

    # Generation configuration arguments
    generation_config: Optional[Dict] = field(default_factory=lambda: DEFAULT_GENERATION_CONFIG | {})

    # Evaluation-related arguments
    eval_type: str = EvalType.CHECKPOINT
    eval_backend: str = EvalBackend.NATIVE
    eval_config: Union[str, Dict, None] = None
    stage: str = EvalStage.ALL
    limit: Optional[int] = None

    # Cache and working directory arguments
    mem_cache: bool = False  # Deprecated, will be removed in v1.0.0.
    use_cache: Optional[str] = None
    work_dir: str = DEFAULT_WORK_DIR

    # Debug and runtime mode arguments
    debug: bool = False
    dry_run: bool = False
    seed: int = 42

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
        pattern_config = registry_tasks.get(data_pattern)

        custom_config = copy.deepcopy(pattern_config)
        custom_config.update({'datasets': [data_pattern]})
        # custom_config.update({'dataset_hub': 'Local'})  # to support `ModelScope`
        if 'dataset_args' in custom_config:
            if data_pattern not in custom_config:
                custom_config['dataset_args'].update({data_pattern: {}})
        else:
            custom_config.update({'dataset_args': {data_pattern: {}}})

        if dataset_dir is not None:
            custom_config['dataset_args'][data_pattern].update({'local_path': dataset_dir})

        if subset_list is not None:
            custom_config['dataset_args'][data_pattern].update({'subset_list': subset_list})

        registry_tasks.update({name: custom_config})
        logger.info(f'** Registered task: {name} with data pattern: {data_pattern}')

    def to_dict(self):
        # Note: to avoid serialization error for some model instance
        return self.__dict__

    def __str__(self):
        return json.dumps(self.to_dict(), indent=4, default=str, ensure_ascii=False)

    @staticmethod
    def load(custom_model: CustomModel, tasks: List[str]) -> List['TaskConfig']:
        res_list = []
        for task_name in tasks:
            task: dict = registry_tasks.get(task_name, None)
            if task is None:
                logger.error(f'No task found in tasks: {list(registry_tasks.keys())}, got task_name: {task_name}')
                continue

            res = TaskConfig(**task)
            res.model = custom_model

            res_list.append(res)

        return res_list

    @staticmethod
    def list():
        return list(registry_tasks.keys())


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
