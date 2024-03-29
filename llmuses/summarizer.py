# Copyright (c) Alibaba, Inc. and its affiliates.
import json
import os
import glob
from typing import List, Union

from llmuses.config import TaskConfig
from llmuses.constants import OutputsStructure
from llmuses.tools.combine_reports import gen_table
from llmuses.utils import process_outputs_structure, yaml_to_dict
from llmuses.utils.logger import get_logger

logger = get_logger()


class Summarizer:

    @staticmethod
    def get_report(outputs_dir: str) -> List[dict]:
        res_list: list = []

        outputs_structure: dict = process_outputs_structure(outputs_dir, is_make=False)
        reports_dir: str = outputs_structure.get(OutputsStructure.REPORTS_DIR)
        if reports_dir is None:
            raise ValueError(f'No reports directory in {outputs_dir}')

        report_files: list = glob.glob(os.path.join(reports_dir, '*.json'))
        for report_file in report_files:
            with open(report_file, 'r') as f:
                res_list.append(json.load(f))

        # report_table: str = gen_table([reports_dir])
        # logger.info(f'*** Report table ***\n{report_table}')

        return res_list

    @staticmethod
    def get_report_from_cfg(task_cfg: Union[str, TaskConfig, List[TaskConfig]]) -> List[dict]:
        """
        Get report from cfg file.

        Args:
            task_cfg: task cfg file path. refer to llmuses/tasks/eval_qwen-7b-chat_v100.yaml

        Returns:
            list: list of report dict.
            A report dict is overall report on a benchmark for specific model.
        """
        if isinstance(task_cfg, list):
            res_list: list = []
            for task_cfg_item in task_cfg:
                res_list.extend(Summarizer.get_report_from_cfg(task_cfg_item))
            res_list = [json.dumps(item, ensure_ascii=False) for item in res_list]
            res_list = list(set(res_list))
            res_list = [json.loads(item) for item in res_list]
            return res_list

        if isinstance(task_cfg, str):
            task_cfg: dict = yaml_to_dict(task_cfg)
        elif isinstance(task_cfg, TaskConfig):
            task_cfg: dict = task_cfg.to_dict()
        else:
            raise ValueError(f'Invalid task_cfg: {task_cfg}')

        logger.info(f'**Task cfg: {task_cfg}')
        outputs_dir: str = task_cfg.get('outputs')
        outputs_dir: str = os.path.expanduser(outputs_dir)
        if outputs_dir is None:
            raise ValueError(f'No outputs_dir in {task_cfg}')

        return Summarizer.get_report(outputs_dir=outputs_dir)


if __name__ == '__main__':
    cfg_file = 'registry/tasks/eval_qwen-7b-chat_v100.yaml'
    report_list = Summarizer.get_report_from_cfg(cfg_file)

    print(report_list)
