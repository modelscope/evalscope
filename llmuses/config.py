# Copyright (c) Alibaba, Inc. and its affiliates.

import os
import copy
from dataclasses import dataclass, asdict, field
from typing import Optional, List

from llmuses.constants import DEFAULT_ROOT_CACHE_DIR
from llmuses.models.custom import CustomModel
from llmuses.utils import yaml_to_dict
from llmuses.utils.logger import get_logger

logger = get_logger()

cur_path = os.path.dirname(os.path.abspath(__file__))

registry_tasks = {
    'arc': yaml_to_dict(os.path.join(cur_path, 'registry/tasks/arc.yaml')),
    'gsm8k': yaml_to_dict(os.path.join(cur_path, 'registry/tasks/gsm8k.yaml')),
    'mmlu': yaml_to_dict(os.path.join(cur_path, 'registry/tasks/mmlu.yaml')),
    'ceval': yaml_to_dict(os.path.join(cur_path, 'registry/tasks/ceval.yaml')),
    'bbh': yaml_to_dict(os.path.join(cur_path, 'registry/tasks/bbh.yaml')),

    # 'bbh_mini': yaml_to_dict(os.path.join(cur_path, 'registry/tasks/bbh_mini.yaml')),
    # 'mmlu_mini': yaml_to_dict(os.path.join(cur_path, 'registry/tasks/mmlu_mini.yaml')),
    # 'ceval_mini': yaml_to_dict(os.path.join(cur_path, 'registry/tasks/ceval_mini.yaml')),

}


@dataclass
class TaskConfig:
    model_args: Optional[dict] = field(default_factory=dict)
    generation_config: Optional[dict] = field(default_factory=dict)
    dataset_args: Optional[dict] = field(default_factory=dict)
    dry_run: bool = False
    model: CustomModel = None
    eval_type: str = 'custom'
    datasets: list = field(default_factory=list)
    work_dir: str = DEFAULT_ROOT_CACHE_DIR
    outputs: str = None
    mem_cache: bool = False
    use_cache: bool = True
    stage: str = 'all'      # `all` or `infer` or `review`
    dataset_hub: str = 'ModelScope'
    dataset_dir: str = ''
    limit: int = None

    # def __post_init__(self):
    #     self.registry_tasks = {
    #         'arc': yaml_to_dict(os.path.join(cur_path, 'registry/tasks/arc.yaml')),
    #         'gsm8k': yaml_to_dict(os.path.join(cur_path, 'registry/tasks/gsm8k.yaml')),
    #         'mmlu': yaml_to_dict(os.path.join(cur_path, 'registry/tasks/mmlu.yaml')),
    #         'ceval': yaml_to_dict(os.path.join(cur_path, 'registry/tasks/ceval.yaml')),
    #         'bbh': yaml_to_dict(os.path.join(cur_path, 'registry/tasks/bbh.yaml')),
    #
    #         'bbh_mini': yaml_to_dict(os.path.join(cur_path, 'registry/tasks/bbh_mini.yaml')),
    #         'mmlu_mini': yaml_to_dict(os.path.join(cur_path, 'registry/tasks/mmlu_mini.yaml')),
    #         'ceval_mini': yaml_to_dict(os.path.join(cur_path, 'registry/tasks/ceval_mini.yaml')),
    #
    #     }

    def registry(self, name: str, data_pattern: str, dataset_dir: str = None) -> None:
        """
        Register a new task (dataset) for evaluation.

        Args:
            name: str, the dataset name.
            data_pattern: str, the data pattern for the task.
                    e.g. `mmlu`, `ceval`, `gsm8k`, ...
                    refer to task_config.list() for all available datasets.
            dataset_dir: str, the directory to store multiple datasets files. e.g. /path/to/data, 
                then your specific custom dataset directory will be /path/to/data/{name}
        """
        available_datasets = self.list()
        if data_pattern not in available_datasets:
            logger.error(f'No dataset found in available datasets: {available_datasets}, got data_pattern: {data_pattern}')
            return

        # Reuse the existing task config and update the datasets
        pattern_config = registry_tasks.get(data_pattern)

        custom_config = copy.deepcopy(pattern_config)
        custom_config.update({'datasets': [data_pattern]})
        custom_config.update({'dataset_hub': 'Local'})     # TODO: to support `ModelScope`
        if dataset_dir is not None:
            custom_config.update({'dataset_args': {data_pattern: {'local_path': os.path.join(dataset_dir, name)}}})

        registry_tasks.update({name: custom_config})
        logger.info(f'** Registered task: {name} with data pattern: {data_pattern}')

    def to_dict(self):
        # Note: to avoid serialization error for some model instance
        _tmp_model = copy.copy(self.model)
        self.model = None
        res_dict = asdict(self)
        res_dict.update({'model': _tmp_model})
        self.model = _tmp_model

        return res_dict

    def load(self, custom_model: CustomModel, tasks: List[str]) -> List['TaskConfig']:
        res_list = []
        for task_name in tasks:
            task: dict = registry_tasks.get(task_name, None)
            if task is None:
                logger.error(f'No task found in tasks: {self.list()}, got task_name: {task_name}')
                continue

            res = TaskConfig(**task)
            res.model = custom_model
            if res.outputs is None:
                res.outputs = os.path.join(res.work_dir,
                                           'outputs',
                                           f"eval_{'-'.join(tasks)}_{res.model.config['model_id']}_{res.model_args.get('revision', 'default')}")
            res_list.append(res)

        return res_list

    def list(self):
        return list(registry_tasks.keys())


class TempModel(CustomModel):

    def __init__(self, config: dict):
        super().__init__(config=config)

    def predict(self, prompt: str, **kwargs):
        return prompt + ': response'


if __name__ == '__main__':
    model = TempModel(config={'model_id': 'test-swift-dummy-model'})
    task_config = TaskConfig()

    # Register a new task
    task_config.registry(name='arc_swift', data_pattern='arc', dataset_dir='/Users/jason/workspace/work/maas/benchmarks/swift_custom_work')

    import json
    swift_eval_task: List[TaskConfig] = task_config.load(custom_model=model, tasks=['gsm8k', 'arc', 'arc_swift'])
    for item in swift_eval_task:
        print(item.to_dict())
        print()

