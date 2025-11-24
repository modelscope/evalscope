# Copyright (c) Alibaba, Inc. and its affiliates.

import glob
import os
import pandas as pd
from tabulate import tabulate
from typing import Dict, List, Tuple, Union

from evalscope.constants import DataCollection
from evalscope.report.report import Report, Subset
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
            # Skip the collection report file
            if os.path.basename(file_path) == DataCollection.REPORT_NAME:
                continue
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


def weighted_average_from_subsets(
    subset_names: List[str], subset_dict: Dict[str, Subset], new_name: str = ''
) -> Subset:
    """Calculate weighted average for given subsets.

    Args:
        subset_names (List[str]): List of subset names to include in the average.
        subset_dict (Dict[str, Subset]): Dictionary mapping subset names to Subset objects.
        new_name (str): Name for the resulting Subset object.

    Returns:
        Subset: A new Subset object with weighted average score
    """
    total_score = 0
    total_count = 0
    for name in subset_names:
        if name in subset_dict:
            subset = subset_dict[name]
            total_score += subset.score * subset.num
            total_count += subset.num

    weighted_avg = total_score / total_count if total_count > 0 else 0
    return Subset(name=new_name, score=weighted_avg, num=total_count)


def unweighted_average_from_subsets(
    subset_names: List[str], subset_dict: Dict[str, Subset], new_name: str = ''
) -> Subset:
    """Calculate unweighted average for given subsets.

    Args:
        subset_names (List[str]): List of subset names to include in the average.
        subset_dict (Dict[str, Subset]): Dictionary mapping subset names to Subset objects.
        new_name (str): Name for the resulting Subset object.

    Returns:
        Subset: A new Subset object with unweighted average score
    """
    scores = []
    total_count = 0
    for name in subset_names:
        if name in subset_dict:
            subset = subset_dict[name]
            scores.append(subset.score)
            total_count += subset.num

    unweighted_avg = sum(scores) / len(scores) if scores else 0
    return Subset(name=new_name, score=unweighted_avg, num=total_count)


def percentage_weighted_average_from_subsets(
    subset_names: List[str], subset_dict: Dict[str, Subset], weights: List[float], new_name: str = ''
) -> Subset:
    """Calculate percentage weighted average for given subsets.

    Args:
        subset_names (List[str]): List of subset names to include in the average.
        subset_dict (Dict[str, Subset]): Dictionary mapping subset names to Subset objects.
        weights (List[float]): The weight for each corresponding accuracy entry.
            Can sum to any positive value – they will be normalised internally.
        new_name (str): Name for the resulting Subset object.

    Returns:
        Subset: A new Subset object with percentage weighted average score.
    """
    assert len(subset_names) == len(weights), \
        'The number of subset names must match the number of weights.'

    valid_subsets = []
    valid_weights = []
    total_count = 0

    for name, weight in zip(subset_names, weights):
        if name in subset_dict:
            subset = subset_dict[name]
            valid_subsets.append(subset)
            valid_weights.append(weight)
            total_count += subset.num

    if not valid_subsets:
        return Subset(name=new_name, score=0, num=0)

    weight_sum = sum(valid_weights)
    assert weight_sum > 0, \
        f"Sum of weights for percentage_weighted_average_from_subsets for '{new_name}' is not positive."

    # Normalise weights so that they sum to 1.0
    weights_norm = [w / weight_sum for w in valid_weights]

    total_score = 0
    for subset, weight in zip(valid_subsets, weights_norm):
        total_score += subset.score * weight

    return Subset(name=new_name, score=total_score, num=total_count)
