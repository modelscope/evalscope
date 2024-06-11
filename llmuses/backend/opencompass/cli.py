# Copyright (c) Alibaba, Inc. and its affiliates.
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Optional, Union

from llmuses.backend.base import BackendArgsParser
from opencompass.cli.cli_arguments import Arguments as OpenCompassArguments


class CmdMode(Enum):

    # The basic mode is to run the command directly,
    # e.g. `python -m run --models model1 model2 --datasets dataset1 dataset2`
    BASIC = 'basic'

    # The script mode is to run the command with a script,
    # e.g. `python -m run your_config_script.py`
    SCRIPT = 'script'


class OpenCompassBackendArgsParser(BackendArgsParser):

    def __init__(self, config: Union[str, dict], **kwargs):
        super().__init__(config, **kwargs)
        self.args = OpenCompassArguments(**self.config_d)

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
            assert self.args.datasets, 'The datasets are required.'
            assert self.args.models, 'The models are required.'

            cmd_str = f'python -m run_oc ' \
                      f'--models {" ".join(self.args.models)} ' \
                      f'--datasets {" ".join(self.args.datasets)} ' \
                      f'{self.get_restore_arg("dry-run", self.args.dry_run)} ' \
                      f'{self.get_arg_with_default("work-dir", self.args.work_dir)}'

        elif cmd_mode == CmdMode.SCRIPT:
            assert self.args.config, 'The script file is required.'
            cmd_str = f'python -m run_oc {self.args.config}'
        else:
            raise ValueError(f'Unsupported command mode: {cmd_mode}')

        return cmd_str


if __name__ == '__main__':

    oc_task_cfg_file = '../../examples/tasks/eval_qwen_oc_cfg.yaml'
    ocp = OpenCompassBackendArgsParser(config=oc_task_cfg_file)
    print(ocp.args)
    print()
    print(ocp.cmd)

