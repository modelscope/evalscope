# Copyright (c) Alibaba, Inc. and its affiliates.
import glob
import json
import os
from typing import List, Union

from evalscope.config import TaskConfig, parse_task_config
from evalscope.constants import EvalBackend
from evalscope.report import gen_table
from evalscope.utils import csv_to_list, get_latest_folder_path
from evalscope.utils.io_utils import OutputsStructure, json_to_dict, yaml_to_dict
from evalscope.utils.logger import get_logger

logger = get_logger()


class Summarizer:

    @staticmethod
    def get_report(outputs_dir: str) -> List[dict]:
        res_list: list = []

        outputs_structure = OutputsStructure(outputs_dir, is_make=False)
        reports_dir: str = outputs_structure.reports_dir
        if reports_dir is None:
            raise ValueError(f'No reports directory in {outputs_dir}')

        report_files: list = glob.glob(os.path.join(reports_dir, '**/*.json'))
        for report_file in report_files:
            with open(report_file, 'r') as f:
                res_list.append(json.load(f))

        report_table: str = gen_table([reports_dir])
        logger.info(f'*** Report table ***\n{report_table}')

        return res_list

    @staticmethod
    def get_report_from_cfg(task_cfg: Union[str, List[str], TaskConfig, List[TaskConfig], dict]) -> List[dict]:
        """
        Get report from cfg file.

        Args:
            task_cfg: task cfg file path. refer to evalscope/tasks/eval_qwen-7b-chat_v100.yaml

        Returns:
            list: list of report dict.
            A report dict is overall report on a benchmark for specific model.
        """
        final_res_list: List[dict] = []
        candidate_task_cfgs: List[TaskConfig] = []

        if isinstance(task_cfg, list):
            for task_cfg_item in task_cfg:
                candidate_task_cfgs.append(parse_task_config(task_cfg_item))
        else:
            candidate_task_cfgs.append(parse_task_config(task_cfg))

        for candidate_task in candidate_task_cfgs:
            logger.info(f'**Loading task cfg for summarizer: {candidate_task}')
            eval_backend = candidate_task.eval_backend

            if eval_backend == EvalBackend.NATIVE:
                outputs_dir: str = os.path.expanduser(candidate_task.work_dir)
                if outputs_dir is None:
                    raise ValueError(f'No outputs_dir in {task_cfg}')
                res_list: list = Summarizer.get_report(outputs_dir=outputs_dir)
                final_res_list.extend(res_list)

            elif eval_backend == EvalBackend.OPEN_COMPASS:
                eval_config = Summarizer.parse_eval_config(candidate_task)

                work_dir = eval_config.get('work_dir') or 'outputs/default'
                if not os.path.exists(work_dir):
                    raise ValueError(f'work_dir {work_dir} does not exist.')

                res_folder_path = get_latest_folder_path(work_dir=work_dir)
                summary_files = glob.glob(os.path.join(res_folder_path, 'summary', '*.csv'))
                if len(summary_files) == 0:
                    raise ValueError(f'No summary files in {res_folder_path}')

                summary_file_path = summary_files[0]
                # Example: [{'dataset': 'gsm8k', 'version': '1d7fe4', 'metric': 'accuracy', 'mode': 'gen', 'qwen-7b-chat': '53.98'} # noqa: E501
                summary_res: List[dict] = csv_to_list(file_path=summary_file_path)
                final_res_list.extend(summary_res)
            elif eval_backend == EvalBackend.VLM_EVAL_KIT:
                eval_config = Summarizer.parse_eval_config(candidate_task)

                work_dir = eval_config.get('work_dir') or 'outputs'
                if not os.path.exists(work_dir):
                    raise ValueError(f'work_dir {work_dir} does not exist.')

                for model in eval_config['model']:
                    if model['name'] == 'CustomAPIModel':
                        model_name = model['type']
                    else:
                        model_name = model['name']

                    csv_files = glob.glob(os.path.join(work_dir, model_name, '*.csv'))
                    json_files = glob.glob(os.path.join(work_dir, model_name, '*.json'))

                    summary_files = csv_files + json_files
                    for summary_file_path in summary_files:
                        if summary_file_path.endswith('csv'):
                            summary_res: dict = csv_to_list(summary_file_path)[0]
                        elif summary_file_path.endswith('json'):
                            summary_res: dict = json_to_dict(summary_file_path)
                        base_name = os.path.basename(summary_file_path)
                        file_name = os.path.splitext(base_name)[0]
                        final_res_list.append({file_name: summary_res})

            elif eval_backend == EvalBackend.THIRD_PARTY:
                raise ValueError('*** The summarizer for Third party evaluation backend is not supported yet ***')
            else:
                raise ValueError(f'Invalid eval_backend: {eval_backend}')

        return final_res_list

    @staticmethod
    def parse_eval_config(candidate_task: TaskConfig):
        eval_config: Union[str, dict] = candidate_task.eval_config
        assert eval_config is not None, 'Please provide eval_config for specific evaluation backend.'

        if isinstance(eval_config, str):
            if eval_config.endswith('.yaml'):
                eval_config: dict = yaml_to_dict(eval_config)
            elif eval_config.endswith('.json'):
                eval_config: dict = json_to_dict(eval_config)
            else:
                raise ValueError(f'Invalid eval_config: {eval_config}')
        return eval_config


if __name__ == '__main__':
    cfg_file = 'registry/tasks/eval_qwen-7b-chat_v100.yaml'
    report_list = Summarizer.get_report_from_cfg(cfg_file)

    print(report_list)
