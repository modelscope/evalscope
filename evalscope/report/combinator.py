# Copyright (c) Alibaba, Inc. and its affiliates.

import glob
import json
import os
from typing import Any, Dict, List, Optional

import pandas as pd
from tabulate import tabulate

from evalscope.api.messages.perf_metrics import PerfSummary
from evalscope.api.metric.semantics import MetricSemantics
from evalscope.constants import DataCollection
from evalscope.metrics.semantics import format_metric_label, format_metric_labels, format_metric_value
from evalscope.report.report import Report, ReportKey, Subset
from evalscope.utils.logger import get_logger

logger = get_logger()
"""
Combine and generate table for reports of LLMs.
"""

_CATEGORY_PLACEHOLDERS = {'', '-', 'default'}


def _format_category_columns_for_display(table: pd.DataFrame) -> None:
    """Hide placeholder category levels and give informative levels user-facing names."""
    category_columns = [column for column in table.columns if column.startswith(ReportKey.category_prefix)]
    informative_columns = []
    for column in category_columns:
        values = table[column].dropna().astype(str).str.strip().str.casefold()
        if values.empty or values.isin(_CATEGORY_PLACEHOLDERS).all():
            table.drop(columns=column, inplace=True)
            continue
        table.loc[table[column].isna() | values.isin(_CATEGORY_PLACEHOLDERS), column] = '-'
        informative_columns.append(column)

    if len(informative_columns) == 1:
        table.rename(columns={informative_columns[0]: ReportKey.category_name}, inplace=True)
        return
    table.rename(
        columns={column: f'{ReportKey.category_name} {index}' for index, column in enumerate(informative_columns, 1)},
        inplace=True,
    )


def _is_report_json(data: Any) -> bool:
    if not isinstance(data, dict):
        return False

    has_report_fields = all(key in data for key in ('name', 'dataset_name', 'model_name'))
    if not has_report_fields:
        return False

    metrics = data.get('metrics')
    perf_metrics = data.get('perf_metrics')
    execution_summary = data.get('execution_summary')
    return (
        (isinstance(metrics, list) and len(metrics) > 0)
        or isinstance(perf_metrics, dict)
        or isinstance(execution_summary, dict)
    )


def get_report_list(reports_path_list: List[str]) -> List[Report]:
    report_list: List[Report] = []
    # Iterate over each report path
    for report_path in reports_path_list:
        model_report_dir = os.path.normpath(report_path)
        report_files = glob.glob(os.path.join(model_report_dir, '**', '*.json'), recursive=True)
        # Iterate over each report file
        for file_path in report_files:
            # Skip the collection report file
            basename = os.path.basename(file_path)
            if basename == DataCollection.REPORT_NAME:
                continue
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                if not _is_report_json(data):
                    continue
                report = Report.from_dict(data)
                report_list.append(report)
            except Exception as e:
                logger.error(f'Error loading report from {file_path}: {e}')
    report_list = sorted(report_list, key=lambda x: (x.model_name, x.dataset_name))
    return report_list


def get_data_frame(
    report_list: List[Report],
    flatten_metrics: bool = True,
    flatten_categories: bool = True,
    add_overall_metric: bool = False,
) -> pd.DataFrame:
    tables = []
    for report in report_list:
        df = report.to_dataframe(
            flatten_metrics=flatten_metrics,
            flatten_categories=flatten_categories,
            add_overall_metric=add_overall_metric,
        )
        tables.append(df)
    return pd.concat(tables, ignore_index=True)


def get_display_data_frame(
    report_list: List[Report],
    flatten_metrics: bool = True,
    flatten_categories: bool = True,
    add_overall_metric: bool = False,
) -> pd.DataFrame:
    """Build a display-only DataFrame with semantic metric labels and formatted values.

    The raw :func:`get_data_frame` contract intentionally remains numeric for downstream
    analysis. Renderers should use this helper so the CLI and service tables share one display
    path without changing the report payload.
    """
    display_table = get_data_frame(
        report_list,
        flatten_metrics=flatten_metrics,
        flatten_categories=flatten_categories,
        add_overall_metric=add_overall_metric,
    ).copy()
    semantics_by_metric = {}
    labels_by_metric = {}
    dataset_labels = {}
    for report in report_list:
        dataset_labels[(report.model_name, report.dataset_name)] = report.dataset_pretty_name or report.dataset_name
        labels = format_metric_labels((metric.identity, metric.semantics) for metric in report.metrics)
        for metric in report.metrics:
            key = (report.model_name, report.dataset_name, metric.name)
            semantics_by_metric[key] = metric.semantics
            labels_by_metric[key] = (
                format_metric_label(metric.identity, metric.semantics, metric.legacy_name)
                if metric.legacy_name
                else labels[metric.identity.key]
            )

    if {'Model', 'Dataset', 'Metric', 'Score'}.issubset(display_table.columns):
        metric_labels = []
        display_scores = []
        for _, row in display_table.iterrows():
            key = (row['Model'], row['Dataset'], row['Metric'])
            metric_labels.append(labels_by_metric.get(key, row['Metric']))
            display_scores.append(format_metric_value(float(row['Score']), semantics_by_metric.get(key)))
        display_table['Metric'] = metric_labels
        display_table['Score'] = display_scores
        display_table['Dataset'] = [
            dataset_labels.get((row['Model'], row['Dataset']), row['Dataset']) for _, row in display_table.iterrows()
        ]

    _format_category_columns_for_display(display_table)
    return display_table


def gen_table(
    reports_path_list: list[str] = None,
    report_list: list[Report] = None,
    flatten_metrics: bool = True,
    flatten_categories: bool = True,
    add_overall_metric: bool = False,
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
        str: A string representation of the table in simple_grid format.

    Raises:
        AssertionError: If neither `reports_path_list` nor `report_list` is provided.
    """
    assert (reports_path_list is not None) or (report_list is not None), (
        'Either reports_path_list or report_list must be provided.'
    )
    if report_list is None:
        report_list = get_report_list(reports_path_list)
    display_table = get_display_data_frame(
        report_list,
        flatten_metrics=flatten_metrics,
        flatten_categories=flatten_categories,
        add_overall_metric=add_overall_metric,
    )

    return tabulate(display_table, headers=display_table.columns, tablefmt='simple_grid', showindex=False)


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
            if subset.num > 0:  # skip subsets with no evaluated samples
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
    assert len(subset_names) == len(weights), 'The number of subset names must match the number of weights.'

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
    assert weight_sum > 0, (
        f"Sum of weights for percentage_weighted_average_from_subsets for '{new_name}' is not positive."
    )

    # Normalise weights so that they sum to 1.0
    weights_norm = [w / weight_sum for w in valid_weights]

    total_score = 0
    for subset, weight in zip(valid_subsets, weights_norm):
        total_score += subset.score * weight

    return Subset(name=new_name, score=total_score, num=total_count)


def gen_perf_table(
    reports_path_list: list[str] = None,
    report_list: list[Report] = None,
) -> Optional[str]:
    """Generate a formatted performance metrics table from reports.

    Extracts ``perf_metrics['summary']`` from each Report and builds a
    per-model × per-dataset table.  Reports that carry no perf data are
    silently skipped.

    Args:
        reports_path_list (list[str], optional): List of directory paths to
            search for report JSON files.  Either this or ``report_list``
            must be provided.
        report_list (list[Report], optional): List of Report objects.
            Either this or ``reports_path_list`` must be provided.

    Returns:
        str: A simple-formatted table string, or ``None`` when no report
        contains perf data.

    Raises:
        AssertionError: If neither argument is provided.
    """
    assert (reports_path_list is not None) or (report_list is not None), (
        'Either reports_path_list or report_list must be provided.'
    )

    if report_list is None:
        report_list = get_report_list(reports_path_list)

    rows = []

    for report in report_list:
        perf = report.perf_metrics
        if not perf:
            continue
        summary = perf.get('summary', {})
        if not summary:
            continue

        ps = PerfSummary.from_dict(summary)
        semantics_map = {
            field_key: MetricSemantics.model_validate(semantics)
            for field_key, semantics in perf.get('metric_semantics', {}).items()
        }

        def display(value: Optional[float], field_key: str) -> str:
            return format_metric_value(value, semantics_map.get(field_key))

        row = {
            'Model': report.model_name,
            'Dataset': report.dataset_pretty_name or report.dataset_name,
            'Num': display(ps.n_samples, 'n_samples'),
            'Avg Lat': display(ps.avg_latency, 'latency'),
            'Avg TTFT': display(ps.avg_ttft, 'ttft'),
            'Avg TPOT': display(ps.avg_tpot, 'tpot'),
            'Avg Thpt': display(ps.avg_output_tps, 'throughput.avg_output_tps'),
            'Avg In': display(ps.avg_input_tokens, 'usage.input_tokens'),
            'Avg Out': display(ps.avg_output_tokens, 'usage.output_tokens'),
        }
        rows.append(row)

    if not rows:
        return None

    df = pd.DataFrame(rows)
    return tabulate(df, headers=df.columns, tablefmt='simple', showindex=False)
