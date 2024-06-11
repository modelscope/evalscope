# Copyright (c) Alibaba, Inc. and its affiliates.
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Optional, Union

from llmuses.backend.base import BackendArgsParser


class CmdMode(Enum):

    # The basic mode is to run the command directly,
    # e.g. `python -m run --models model1 model2 --datasets dataset1 dataset2`
    BASIC = 'basic'

    # The script mode is to run the command with a script,
    # e.g. `python -m run your_config_script.py`
    SCRIPT = 'script'


@dataclass
class OpenCompassArgs:
    models: Optional[list] = field(default_factory=list)
    datasets: Optional[list] = field(default_factory=list)
    dry_run: bool = False
    work_dir: Optional[str] = None

    script_file: Optional[str] = None


class OpenCompassBackendArgsParser(BackendArgsParser):

    def __init__(self, config: Union[str, dict], **kwargs):
        super().__init__(config, **kwargs)

        self.oc_args = OpenCompassArgs()
        self.oc_args.models = self.config_d.get('models', [])
        self.oc_args.datasets = self.config_d.get('datasets', [])
        self.oc_args.dry_run = self.config_d.get('dry_run', False)
        self.oc_args.work_dir = self.config_d.get('work_dir', None)

        self.oc_args.script_file = self.config_d.get('script_file', None)

    @property
    def args(self):
        return self.oc_args

    @property
    def cmd(self):
        return self.get_cmd()

    @staticmethod
    def get_restore_arg(arg_name: str, arg_val: bool):
        if arg_val:
            return f'--{arg_name}'
        else:
            return ''

    @staticmethod
    def get_arg_with_default(arg_name: str, arg_val: Optional[str] = None):
        if arg_val:
            return f'--{arg_name} {arg_val}'
        else:
            return ''

    def get_cmd(self, cmd_mode: str = CmdMode.BASIC):

        if cmd_mode == CmdMode.BASIC:
            cmd_str = f'python -m run_oc ' \
                      f'--models {" ".join(self.oc_args.models)} ' \
                      f'--datasets {" ".join(self.oc_args.datasets)} ' \
                      f'{OpenCompassBackendArgsParser.get_restore_arg("dry-run", self.oc_args.dry_run)} ' \
                      f'{OpenCompassBackendArgsParser.get_arg_with_default("work-dir", self.oc_args.work_dir)}'

        elif cmd_mode == CmdMode.SCRIPT:
            if self.oc_args.script_file is None:
                raise ValueError('The script file is required in script mode.')
            # TODO: command `run` to be replaced with another name
            cmd_str = f'python -m run_oc {self.oc_args.script_file}'
        else:
            raise ValueError(f'Unsupported command mode: {cmd_mode}')

        return cmd_str


if __name__ == '__main__':

    oc_task_cfg_file = '../../examples/tasks/eval_qwen_oc_cfg.yaml'
    ocp = OpenCompassBackendArgsParser(config=oc_task_cfg_file)
    print(ocp.args)
    print()
    print(ocp.cmd)

