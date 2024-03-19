# Copyright (c) Alibaba, Inc. and its affiliates.

import os
from dataclasses import dataclass, asdict, field
from typing import Optional, List

from llmuses.constants import DEFAULT_ROOT_CACHE_DIR
from llmuses.models.custom import CustomModel
from llmuses.utils import yaml_to_dict
from llmuses.utils.logger import get_logger

logger = get_logger()

cur_path = os.path.dirname(os.path.abspath(__file__))


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
    dataset_hub: str = 'ModelScope'
    dataset_dir: str = ''
    limit: int = None

    def __post_init__(self):
        self.registry_tasks = {
            'arc': yaml_to_dict(os.path.join(cur_path, 'registry/tasks/arc.yaml')),
            'gsm8k': yaml_to_dict(os.path.join(cur_path, 'registry/tasks/gsm8k.yaml')),
        }

    def to_dict(self):
        return asdict(self)

    def load(self, custom_model: CustomModel, tasks: List[str]):
        tmp_d: dict = {}
        datasets: list = []
        for task_name in tasks:
            task: dict = self.registry_tasks.get(task_name, None)
            if task is None:
                logger.error(f'No task found in tasks: {self.list()}, got task_name: {task_name}')
                continue
            tmp_d = task
            datasets.extend(task.get('datasets', []))

        tmp_d.update({'datasets': datasets})
        tmp_d.update({'model': custom_model})

        res = TaskConfig(**tmp_d)
        if res.outputs is None:
            res.outputs = os.path.join(res.work_dir,
                                       'outputs',
                                       f"eval_{'-'.join(tasks)}_{res.model.config['model_id']}_{res.model_args.get('revision', 'default')}")

        return res

    def list(self):
        return list(self.registry_tasks.keys())


class TempModel(CustomModel):

    def __init__(self, config: dict):
        super().__init__(config=config)

    def predict(self, prompt: str, **kwargs):
        return prompt + ': response'


if __name__ == '__main__':
    model = TempModel(config={'model_id': 'test-swift-dummy-model'})
    task_cfg = TaskConfig()

    task_inst: TaskConfig = task_cfg.load(custom_model=model, tasks=['arc', 'gsm8k'])

    print(task_inst.to_dict())
    print(task_inst.list())
