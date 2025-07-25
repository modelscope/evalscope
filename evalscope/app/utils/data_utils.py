"""
Data loading and processing utilities for the Evalscope dashboard.
"""
import glob
import numpy as np
import os
import pandas as pd
from typing import Any, Dict, List, Union

from evalscope.constants import DataCollection
from evalscope.report import Report, ReportKey, get_data_frame, get_report_list
from evalscope.utils.io_utils import OutputsStructure, yaml_to_dict
from evalscope.utils.logger import get_logger
from ..constants import DATASET_TOKEN, MODEL_TOKEN, REPORT_TOKEN

logger = get_logger()


def scan_for_report_folders(root_path):
    """Scan for folders containing reports subdirectories"""
    logger.debug(f'Scanning for report folders in {root_path}')
    if not os.path.exists(root_path):
        return []

    reports = []
    # Iterate over all folders in the root path
    for folder in glob.glob(os.path.join(root_path, '*')):
        # Check if reports folder exists
        reports_path = os.path.join(folder, OutputsStructure.REPORTS_DIR)
        if not os.path.exists(reports_path):
            continue

        # Iterate over all items in reports folder
        for model_item in glob.glob(os.path.join(reports_path, '*')):
            if not os.path.isdir(model_item):
                continue
            datasets = []
            for dataset_item in glob.glob(os.path.join(model_item, '*.json')):
                datasets.append(os.path.splitext(os.path.basename(dataset_item))[0])
            datasets = DATASET_TOKEN.join(datasets)
            reports.append(
                f'{os.path.basename(folder)}{REPORT_TOKEN}{os.path.basename(model_item)}{MODEL_TOKEN}{datasets}')

    reports = sorted(reports, reverse=True)
    logger.debug(f'reports: {reports}')
    return reports


def process_report_name(report_name: str):
    prefix, report_name = report_name.split(REPORT_TOKEN)
    model_name, datasets = report_name.split(MODEL_TOKEN)
    datasets = datasets.split(DATASET_TOKEN)
    return prefix, model_name, datasets


def load_single_report(root_path: str, report_name: str):
    prefix, model_name, datasets = process_report_name(report_name)
    report_path_list = os.path.join(root_path, prefix, OutputsStructure.REPORTS_DIR, model_name)
    report_list = get_report_list([report_path_list])

    config_files = glob.glob(os.path.join(root_path, prefix, OutputsStructure.CONFIGS_DIR, '*.yaml'))
    if not config_files:
        raise FileNotFoundError(
            f'No configuration files found in {os.path.join(root_path, prefix, OutputsStructure.CONFIGS_DIR)}')
    task_cfg_path = config_files[0]
    task_cfg = yaml_to_dict(task_cfg_path)
    return report_list, datasets, task_cfg


def load_multi_report(root_path: str, report_names: List[str]):
    report_list = []
    for report_name in report_names:
        prefix, model_name, datasets = process_report_name(report_name)
        report_path_list = os.path.join(root_path, prefix, OutputsStructure.REPORTS_DIR, model_name)
        reports = get_report_list([report_path_list])
        report_list.extend(reports)
    return report_list


def get_acc_report_df(report_list: List[Report]):
    data_dict = []
    for report in report_list:
        if report.name == DataCollection.NAME:
            for metric in report.metrics:
                for category in metric.categories:
                    item = {
                        ReportKey.model_name: report.model_name,
                        ReportKey.dataset_name: '/'.join(category.name),
                        ReportKey.score: category.score,
                        ReportKey.num: category.num,
                    }
                    data_dict.append(item)
        else:
            item = {
                ReportKey.model_name: report.model_name,
                ReportKey.dataset_name: report.dataset_name,
                ReportKey.score: report.score,
                ReportKey.num: report.metrics[0].num,
            }
            data_dict.append(item)
    df = pd.DataFrame.from_dict(data_dict, orient='columns')

    styler = style_df(df, columns=[ReportKey.score])
    return df, styler


def style_df(df: pd.DataFrame, columns: List[str] = None):
    # Apply background gradient to the specified columns
    styler = df.style.background_gradient(subset=columns, cmap='RdYlGn', vmin=0.0, vmax=1.0, axis=0)
    # Format the dataframe with a precision of 4 decimal places
    styler.format(precision=4)
    return styler


def get_compare_report_df(acc_df: pd.DataFrame):
    df = acc_df.pivot_table(index=ReportKey.model_name, columns=ReportKey.dataset_name, values=ReportKey.score)
    df.reset_index(inplace=True)

    styler = style_df(df)
    return df, styler


def get_single_dataset_df(df: pd.DataFrame, dataset_name: str):
    df = df[df[ReportKey.dataset_name] == dataset_name]
    styler = style_df(df, columns=[ReportKey.score])
    return df, styler


def get_report_analysis(report_list: List[Report], dataset_name: str) -> str:
    for report in report_list:
        if report.dataset_name == dataset_name:
            return report.analysis
    return 'N/A'


def get_model_prediction(work_dir: str, model_name: str, dataset_name: str, subset_name: str):
    data_path = os.path.join(work_dir, OutputsStructure.REVIEWS_DIR, model_name)
    subset_name = subset_name.replace('/', '_')  # for collection report
    review_path = os.path.join(data_path, f'{dataset_name}_{subset_name}.jsonl')
    logger.debug(f'review_path: {review_path}')
    origin_df = pd.read_json(review_path, lines=True)

    ds = []
    for i, item in origin_df.iterrows():
        raw_input = item['raw_input']
        sample_index = item['index']
        for choice_index, choice in enumerate(item['choices']):
            raw_pred_answer = choice['message']['content']
            parsed_gold_answer = choice['review']['gold']
            parsed_pred_answer = choice['review']['pred']
            score = choice['review']['result']
            raw_d = {
                'Index': f'{sample_index}_{choice_index}',
                'Input': raw_input,
                'Generated': raw_pred_answer if raw_pred_answer != parsed_pred_answer else '*Same as Pred*',
                'Gold': parsed_gold_answer if parsed_gold_answer != raw_input else '*Same as Input*',
                'Pred': parsed_pred_answer,
                'Score': score,
                'NScore': normalize_score(score)
            }
            ds.append(raw_d)

    df_subset = pd.DataFrame(ds)
    return df_subset


def normalize_score(score):
    try:
        if isinstance(score, bool):
            return 1.0 if score else 0.0
        elif isinstance(score, dict):
            for key in score:
                return float(score[key])
            return 0.0
        else:
            return float(score)
    except (ValueError, TypeError):
        return 0.0
