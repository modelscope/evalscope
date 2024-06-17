# Copyright (c) Alibaba, Inc. and its affiliates.
from enum import Enum
from typing import Optional, Union
import subprocess

from llmuses.utils import is_module_installed, get_module_path, get_valid_list
from llmuses.backend.base import BackendManager
from llmuses.utils.logger import get_logger

logger = get_logger()


class CmdMode(Enum):

    # The basic mode is to run the command directly,
    # e.g. `python -m run --models model1 model2 --datasets dataset1 dataset2`
    BASIC = 'basic'

    # The script mode is to run the command with a script,
    # e.g. `python -m run your_config_script.py`
    SCRIPT = 'script'


class RunMode(Enum):

    # The command mode is to run the command directly with the command line.
    CMD = 'cmd'

    # The function mode is to run the command with a function call -- run_task().
    FUNCTION = 'function'


class OpenCompassBackendManager(BackendManager):

    def __init__(self, config: Union[str, dict], **kwargs):
        """
        The backend manager for OpenCompass.

        Args:
            config: Union[str, dict], the configuration yaml-file or the configuration dictionary.
                attributes:
                    datasets: list, the datasets.
                    models: list, the models.
                    work_dir (Optional): str, the working directory. Default to None, which means the current directory.
                    dry_run (Optional): bool, the dry-run flag. Default to False.
                    debug (Optional): bool, the debug flag. Default to False.
                    reuse (Optional): str, reuse previous outputs & results. Default to None.
                    generation_kwargs (Optional): dict, the generation config. Default to {}.

                example:
                    # TODO: add demo config
                    config = dict(
                        datasets=[mmlu, ceval],
                        models=[...],
                        ...
                    )

            **kwargs: the keyword arguments.
        """

        self._check_env()
        super().__init__(config, **kwargs)

        from opencompass.cli.arguments import Arguments as OpenCompassArguments
        self.args = OpenCompassArguments(**self.config_d)

    @property
    def cmd(self):
        return self.get_cmd()

    @staticmethod
    def _check_env():
        if is_module_installed('opencompass'):
            logger.info('Please make sure you have installed the `ms-opencompass`: `pip install ms-opencompass`')
        else:
            raise ModuleNotFoundError('Please install the `ms-opencompass` first: `pip install ms-opencompass`')

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

    def load_task_template(self):
        """
        Load the initial OpenCompass task template from task config file.

        Returns:
            (mmengine.config.config.Config), the initial task template config.
        """
        from opencompass.utils.run import get_config_from_arg

        template_config_path = get_module_path('llmuses.backend.opencompass.tasks.eval_api')
        self.args.config = template_config_path
        return get_config_from_arg(self.args)

    @staticmethod
    def list_datasets():
        from opencompass.utils.run import get_config_from_arg
        from dataclasses import dataclass

        @dataclass
        class TempArgs:
            config: str
            accelerator: str = None

        template_config_path = get_module_path('llmuses.backend.opencompass.tasks.eval_api')
        template_cfg = get_config_from_arg(TempArgs(config=template_config_path))

        return list(set([_dataset['dataset_name'] for _dataset in template_cfg.datasets]))

    def get_task_args(self):
        return self.args

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

    def run(self, run_mode: str = RunMode.FUNCTION):
        """
        The entry function to run the OpenCompass task.

        Args:
            run_mode: str, the running mode, e.g. 'function' or 'cmd'.

        Returns:
            None
        """
        if run_mode == RunMode.FUNCTION:
            from opencompass.cli.main import run_task

            # Get valid datasets
            datasets = self.args.datasets
            datasets_all = self.list_datasets()
            if not datasets:
                logger.warning(f'No datasets are specified in the config. Use all the datasets: {datasets_all}')
                datasets = datasets_all
            else:
                valid_datasets, invalid_datasets = get_valid_list(datasets, datasets_all)
                if len(invalid_datasets) > 0:
                    logger.error(f'Invalid datasets: {invalid_datasets}, '
                                 f'refer to the following list to get proper dataset name: {datasets_all}')
                assert len(valid_datasets) > 0, f'No valid datasets. ' \
                                                f'To get the valid datasets, please refer to {datasets_all}'
                datasets = valid_datasets

            # Get valid models
            # TODO
            models = self.args.models
            ...

            # Load the initial task template and override configs
            template_cfg = self.load_task_template()
            template_cfg.datasets = datasets
            template_cfg.models = models

            print(template_cfg.datasets)
            ...

            # create tmp_config_file and config.dump(tmp_config_file) and self.args.config = tmp_config_file

            run_task(self.args)

        # TODO: add more arguments for the command line
        elif run_mode == RunMode.CMD:
            subprocess.run(self.cmd, text=True, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        else:
            raise ValueError(f'Unsupported run mode: {run_mode}')


if __name__ == '__main__':
    pass
    # TODO
    # 1. 使用mmengine config获取eval_api.py作为initial task template
    # 2. 根据传入的datasets做过滤(使用prefix_filter)，然后赋值给 config.datasets
    # 3. 组装models，然后赋值给 config.models
    # 4. 创建临时文件，并将config dump到临时文件中

    # 其它：1）增加list_datasets()方法，用于列出所有的datasets；2）增加list_models()方法，用于列出所有的models

    ocm = OpenCompassBackendManager(config={'datasets': ['mmlu', 'ceval', 'xxx']})
    ocm.run()

    # oc_task_cfg_file = '../../examples/tasks/eval_qwen_oc_cfg.yaml'
    # ocm = OpenCompassBackendManager(config=oc_task_cfg_file)
    # print(ocm.args)
    # print()
    # print(ocm.cmd)

    #
    # from dataclasses import dataclass
    # from opencompass.utils.run import get_config_from_arg
    #
    # @dataclass
    # class TempArgs:
    #     config: str
    #     accelerator: str = None
    #
    # args = TempArgs(
    #     config='/Users/jason/workspace/work/maas/github/opencompass/configs/eval_openai_format_task_teval_v2_qwen.py')
    #
    # config = get_config_from_arg(args)   # mmengine.config.config.Config
    # # print(config)
    #
    # tmp_config_file = '/Users/jason/workspace/work/maas/github/llmuses_work/llmuses/temp/configs/temp_config.py'
    # config.dump(tmp_config_file)
    # new_config = get_config_from_arg(TempArgs(config=tmp_config_file))
    # print(f'>>new config: {new_config}')
    # print(f'>>new models: {new_config.models}')

