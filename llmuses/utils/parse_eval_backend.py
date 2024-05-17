# Copyright (c) Alibaba, Inc. and its affiliates.
from dataclasses import dataclass, field, asdict
from typing import Optional, Union

from llmuses.utils import yaml_to_dict


class BackendArgsParser:
    def __init__(self, config: Union[str, dict], **kwargs):
        """
        BackendParser for parsing the evaluation backend configuration.
        config: str or dict, the configuration of the evaluation backend.
            could be a string of the path to the configuration file (yaml), or a dictionary.
        """
        if isinstance(config, str):
            self.config_d = yaml_to_dict(config)
        else:
            self.config_d = config

        self.kwargs = kwargs


@dataclass
class OpenCompassArgs:
    models: Optional[list] = field(default_factory=list)
    datasets: Optional[list] = field(default_factory=list)
    dry_run: bool = False


class OpenCompassBackendArgsParser(BackendArgsParser):

    def __init__(self, config: Union[str, dict], **kwargs):
        super().__init__(config, **kwargs)

        self.oc_args = OpenCompassArgs()
        self.oc_args.models = self.config_d.get('models', [])
        self.oc_args.datasets = self.config_d.get('datasets', [])
        self.oc_args.dry_run = self.config_d.get('dry_run', False)

    @property
    def args(self):
        return self.oc_args

    @property
    def cmd(self):
        return self.get_cmd()

    @staticmethod
    def get_restore_arg(arg_name: str, arg_val: bool):
        if arg_val:
            return arg_name
        else:
            return ''

    def get_cmd(self):
        cmd_str = f'python3 -m run ' \
                  f'--models {" ".join(self.oc_args.models)} ' \
                  f'--datasets {" ".join(self.oc_args.datasets)} ' \
                  f'{OpenCompassBackendArgsParser.get_restore_arg("--dry_run", self.oc_args.dry_run)}'

        return cmd_str


if __name__ == '__main__':

    oc_task_cfg_file = '/Users/jason/workspace/work/maas/github/llmuses_work/llmuses/temp/configs/qwen-1p5-7b-chat_mmlu_cfg.yaml'
    ocp = OpenCompassBackendArgsParser(config=oc_task_cfg_file)
    print(ocp.args)
    print(ocp.cmd)

