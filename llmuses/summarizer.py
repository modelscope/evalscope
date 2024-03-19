# Copyright (c) Alibaba, Inc. and its affiliates.
import json
import os
import glob
from typing import List

from llmuses.constants import OutputsStructure
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

        return res_list

    @staticmethod
    def get_report_from_cfg(cfg_file: str) -> List[dict]:
        """
        Get report from cfg file.

        Args:
            cfg_file: task cfg file path. refer to llmuses/tasks/eval_qwen-7b-chat_v100.yaml

        Returns:
            list: list of report dict.
            A report dict is a overall report on a benchmark for specific model.
        """
        task_cfg: dict = yaml_to_dict(cfg_file)
        logger.info(f'**Task cfg: {task_cfg}')
        outputs_dir: str = task_cfg.get('outputs')
        if outputs_dir is None:
            raise ValueError(f'No outputs_dir in {cfg_file}')

        return Summarizer.get_report(outputs_dir=outputs_dir)


if __name__ == '__main__':
    cfg_file = 'registry/tasks/eval_qwen-7b-chat_v100.yaml'
    report_list = Summarizer.get_report_from_cfg(cfg_file)

    print(report_list)
