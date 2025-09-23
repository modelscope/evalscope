# Copyright (c) Alibaba, Inc. and its affiliates.

import glob
import os
import pandas as pd
from tabulate import tabulate
from typing import List, Tuple

from evalscope.report.report import Report
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


def get_data_frame(
    report_list: List[Report],
    flatten_metrics: bool = True,
    flatten_categories: bool = True,
    add_overall_metric: bool = False
) -> pd.DataFrame:
    tables = []
    for report in report_list:
        df = report.to_dataframe(
            flatten_metrics=flatten_metrics,
            flatten_categories=flatten_categories,
            add_overall_metric=add_overall_metric
        )
        tables.append(df)
    return pd.concat(tables, ignore_index=True)


def gen_table(
    reports_path_list: list[str] = None,
    report_list: list[Report] = None,
    flatten_metrics: bool = True,
    flatten_categories: bool = True,
    add_overall_metric: bool = False
) -> str:
    """
    Generates a formatted table from a list of report paths or Report objects.

    Args:
        reports_path_list (list[str], optional): List of file paths to report files.
            Either this or `report_list` must be provided.
        report_list (list[Report], optional): List of Report objects.
            Either this or `reports_path_list` must be provided.
        flatten_metrics (bool, optional): Whether to flatten the metrics in the output table. Defaults to True.
        flatten_categories (bool, optional): Whether to flatten the categories in the output table. Defaults to True.
        add_overall_metric (bool, optional): Whether to add an overall metric column to the table. Defaults to False.

    Returns:
        str: A string representation of the table in grid format.

    Raises:
        AssertionError: If neither `reports_path_list` nor `report_list` is provided.
    """
    assert (reports_path_list is not None) or (report_list is not None), \
        'Either reports_path_list or report_list must be provided.'
    if report_list is None:
        report_list = get_report_list(reports_path_list)
    # Generate a DataFrame from the report list
    table = get_data_frame(
        report_list,
        flatten_metrics=flatten_metrics,
        flatten_categories=flatten_categories,
        add_overall_metric=add_overall_metric
    )
    return tabulate(table, headers=table.columns, tablefmt='grid', showindex=False)
