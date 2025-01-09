# Copyright (c) Alibaba, Inc. and its affiliates.

import glob
import os
import pandas as pd
from tabulate import tabulate
from typing import List

from evalscope.report.utils import Report
from evalscope.utils.logger import get_logger

logger = get_logger()
"""
Combine and generate table for reports of LLMs.
"""


def get_model_reports(model_report_dir: str) -> List[Report]:
    model_report_dir = os.path.normpath(model_report_dir)
    report_files = glob.glob(os.path.join(model_report_dir, '**', '*.json'), recursive=True)
    report_list = []
    for file_path in report_files:
        try:
            report = Report.from_json(file_path)
            report_list.append(report)
        except Exception as e:
            logger.error(f"Error loading report from {file_path}: {e}")
    return report_list


def _get_data(reports_path_list):
    report_list: List[Report] = []
    for report_path in reports_path_list:
        report_list.extend(get_model_reports(report_path))

    report_list = sorted(report_list, key=lambda x: (x.model_name, x.dataset_name))

    headers = ['Model Name', 'Dataset Name', 'Metric Name', 'Category Name', 'Subset Name', 'Num', 'Score']
    table = []

    for report in report_list:
        for metric in report.metrics:
            for category in metric.categories:
                for subset in category.subsets:
                    table.append([
                        report.model_name, report.dataset_name, metric.name, category.name, subset.name, subset.num,
                        subset.score
                    ])
    return report_list, table, headers


def gen_table(reports_path_list: list) -> str:
    report_list, table, headers = _get_data(reports_path_list)
    return tabulate(table, headers, tablefmt='grid')


def gen_data_frame(reports_path_list: list) -> pd.DataFrame:
    report_list, table, headers = _get_data(reports_path_list)
    return report_list, pd.DataFrame(table, columns=headers)


class ReportsRecorder:
    COMMON_DATASET_PATH = []
    CUSTOM_DATASET_PATH = []

    def __init__(self, oss_url: str = '', endpoint: str = ''):
        pass


if __name__ == '__main__':
    report_dir_1 = '/mnt/data/data/user/maoyunlin.myl/eval-scope/outputs/20250107_204445/reports/Qwen2-0.5B-Instruct'
    report_dir_2 = '/mnt/data/data/user/maoyunlin.myl/eval-scope/outputs/20250107_204445/reports'

    report_table = gen_table([report_dir_1, report_dir_2])
    print(report_table)

    # ALL VALUES ONLY FOR EXAMPLE
    # +--------------------------+-------------------+-------------+
    # | Model                    | CompetitionMath   | GSM8K       |
    # +==========================+===================+=============+
    # | ZhipuAI_chatglm2-6b-base | 25.0 (acc)        | 30.50 (acc) |
    # +--------------------------+-------------------+-------------+
    # | ZhipuAI_chatglm2-6b      | 30.5 (acc)        | 40.50 (acc) |
    # +--------------------------+-------------------+-------------+
