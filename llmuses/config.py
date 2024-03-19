# Copyright (c) Alibaba, Inc. and its affiliates.

from dataclasses import dataclass, asdict, field
from typing import Optional

from llmuses.models.custom import CustomModel
from llmuses.utils import yaml_to_dict


@dataclass
class TaskConfig:
    model_args: Optional[dict] = field(default_factory=dict)
    generation_config: Optional[dict] = field(default_factory=dict)
    dataset_args: Optional[dict] = field(default_factory=dict)
    dry_run: bool = False
    model: CustomModel = None
    eval_type: str = 'custom'
    datasets: list = field(default_factory=list)
    work_dir: str = ''
    outputs: str = ''
    mem_cache: bool = False
    dataset_hub: str = 'ModelScope'
    dataset_dir: str = ''
    limit: int = None

    def __post_init__(self):
        self.registry_tasks = {
            'arc': yaml_to_dict('registry/tasks/arc.yaml'),
        }

    def to_dict(self):
        return asdict(self)

    def load(self, task_name: str):
        task: dict = self.registry_tasks.get(task_name, None)
        if task is None:
            raise ValueError(f'No task found in tasks: {self.list()}, got task_name: {task_name}')

        return TaskConfig(**task)

    def list(self):
        return list(self.registry_tasks.keys())


class TempModel(CustomModel):

    def __init__(self, config: dict):
        super().__init__(config=config)

    def predict(self, prompt: str, **kwargs):
        return prompt + ': response'


if __name__ == '__main__':
    model = TempModel(config={'model_id': 'test_model'})
    task_cfg = TaskConfig(model=model)

    task_arc: TaskConfig = task_cfg.load('arc')

    print(task_arc.datasets)
