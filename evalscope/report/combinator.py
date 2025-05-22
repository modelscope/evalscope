# Copyright (c) Alibaba, Inc. and its affiliates.

import glob
import os
import pandas as pd
from tabulate import tabulate
from typing import List, Tuple

from evalscope.report.utils import Report
from evalscope.utils.logger import get_logger

logger = get_logger()
"""
Combine and generate table for reports of LLMs.
"""


def get_report_list(reports_path_list: List[str]) -> List[Report]:
    report_list: List[Report] = []
    # Iterate over each report path
    for report_path in reports_path_list:
        model_report_dir = os.path.normpath(report_path)
        report_files = glob.glob(os.path.join(model_report_dir, '**', '*.json'), recursive=True)
        # Iterate over each report file
        for file_path in report_files:
            try:
                report = Report.from_json(file_path)
                report_list.append(report)
            except Exception as e:
                logger.error(f'Error loading report from {file_path}: {e}')
    report_list = sorted(report_list, key=lambda x: (x.model_name, x.dataset_name))
    return report_list


def get_data_frame(report_list: List[Report],
                   flatten_metrics: bool = True,
                   flatten_categories: bool = True) -> pd.DataFrame:
    tables = []
    for report in report_list:
        df = report.to_dataframe(flatten_metrics=flatten_metrics, flatten_categories=flatten_categories)
        tables.append(df)
    return pd.concat(tables, ignore_index=True)


def gen_table(reports_path_list: list) -> str:
    report_list = get_report_list(reports_path_list)
    table = get_data_frame(report_list)
    return tabulate(table, headers=table.columns, tablefmt='grid', showindex=False)


def gen_report_table(report: Report) -> str:
    """
    Generate a report table for a single report.
    """
    table = report.to_dataframe(flatten_metrics=True, flatten_categories=True)
    return tabulate(table, headers=table.columns, tablefmt='grid', showindex=False)


class ReportsRecorder:
    COMMON_DATASET_PATH = []
    CUSTOM_DATASET_PATH = []

    def __init__(self, oss_url: str = '', endpoint: str = ''):
        pass


if __name__ == '__main__':
    report_dir_1 = './outputs/20250117_151926'
    # report_dir_2 = './outputs/20250107_204445/reports'

    report_table = gen_table([report_dir_1])
    print(report_table)

    # ALL VALUES ONLY FOR EXAMPLE
    # +--------------------------+-------------------+-------------+
    # | Model                    | CompetitionMath   | GSM8K       |
    # +==========================+===================+=============+
    # | ZhipuAI_chatglm2-6b-base | 25.0 (acc)        | 30.50 (acc) |
    # +--------------------------+-------------------+-------------+
    # | ZhipuAI_chatglm2-6b      | 30.5 (acc)        | 40.50 (acc) |
    # +--------------------------+-------------------+-------------+
