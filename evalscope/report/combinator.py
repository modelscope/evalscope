# Copyright (c) Alibaba, Inc. and its affiliates.

import glob
import os
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
    report_files = glob.glob(os.path.join(model_report_dir, '**/*.json'))

    report_list = []
    for file_path in report_files:
        report = Report.from_json(file_path)
        report_list.append(report)
    return report_list


def gen_table(reports_path_list: list) -> str:
    report_list: List[Report] = []
    for report_path in reports_path_list:
        report_list.extend(get_model_reports(report_path))

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

    return tabulate(table, headers, tablefmt='grid')


class ReportsRecorder:
    COMMON_DATASET_PATH = []
    CUSTOM_DATASET_PATH = []

    def __init__(self, oss_url: str = '', endpoint: str = ''):
        pass


if __name__ == '__main__':
    report_dir_1 = '/to/path/20231129_020533_default_ZhipuAI_chatglm2-6b-base_none/reports'
    report_dir_2 = '/to/path/20231129_020533_default_ZhipuAI_chatglm2-6b_none/reports'

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
