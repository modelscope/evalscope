import glob
import gradio as gr
import logging
import numpy as np
import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dataclasses import dataclass
from typing import Any, List, Union

from evalscope.report import Report, ReportKey, gen_data_frame
from evalscope.utils.io_utils import OutputsStructure, yaml_to_dict
from evalscope.utils.logger import get_logger

logger = get_logger(log_level=logging.DEBUG, force=True)


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
                datasets.append(os.path.basename(dataset_item).split('.')[0])
            datasets = ','.join(datasets)
            reports.append(f'{os.path.basename(folder)}@{os.path.basename(model_item)}:{datasets}')

    reports = sorted(reports, reverse=True)
    logger.debug(f'reports: {reports}')
    return reports


def load_single_report(root_path: str, report_path: str):
    prefix, report_name = report_path.split('@')
    model_name, datasets = report_name.split(':')
    datasets = datasets.split(',')
    report_path_list = os.path.join(root_path, prefix, OutputsStructure.REPORTS_DIR, model_name)
    report_list, data_frame = gen_data_frame([report_path_list])

    task_cfg_path = glob.glob(os.path.join(root_path, prefix, OutputsStructure.CONFIGS_DIR, '*.yaml'))[0]
    task_cfg = yaml_to_dict(task_cfg_path)
    return report_list, data_frame, task_cfg


def get_single_report_data(report_list: List[Report]):
    data_dict = []
    for report in report_list:
        data_dict.append({
            ReportKey.model_name: report.model_name,
            ReportKey.dataset_name: report.dataset_name,
            ReportKey.score: report.score,
            ReportKey.num: report.metrics[0].num,
        })
    df = pd.DataFrame.from_dict(data_dict, orient='columns')
    return df


def plot_single_report_scores(df: pd.DataFrame):
    plot = px.bar(
        df,
        x=df[ReportKey.dataset_name],
        y=df[ReportKey.score],
        color=df[ReportKey.dataset_name],
        template='plotly_dark')
    return plot


def plot_single_report_sunburst(df: pd.DataFrame):
    plot = px.sunburst(
        df,
        path=[ReportKey.dataset_name, ReportKey.category_name, ReportKey.subset_name],
        values=ReportKey.num,
        color=ReportKey.score,
        color_continuous_scale='RdBu',
        color_continuous_midpoint=np.average(df[ReportKey.score], weights=df[ReportKey.num]),
        template='plotly_dark')
    plot.update_traces(insidetextorientation='radial')
    plot.update_layout(margin=dict(t=10, l=10, r=10, b=10), coloraxis=dict(cmin=0, cmax=1))
    return plot


def get_single_dataset_data(df: pd.DataFrame, dataset_name: str):
    return df[df[ReportKey.dataset_name] == dataset_name]


def plot_single_dataset_scores(df: pd.DataFrame):
    # TODO: add metric radio and relace category name
    plot = px.bar(
        df,
        x=df[ReportKey.metric_name],
        y=df[ReportKey.score],
        color=df[ReportKey.subset_name],
        template='plotly_dark',
        barmode='group')
    return plot


def dict_to_markdown(data) -> str:
    markdown_lines = []

    for key, value in data.items():
        bold_key = f'**{key}**'

        if isinstance(value, list):
            value_str = '\n' + '\n'.join([f'  - {item}' for item in value])
        elif isinstance(value, dict):
            value_str = dict_to_markdown(value)
        else:
            value_str = str(value)

        value_str = process_string(value_str)
        markdown_line = f'{bold_key}: {value_str}'
        markdown_lines.append(markdown_line)

    return '\n\n'.join(markdown_lines)


def process_string(string: str, max_length: int = 200) -> str:
    if len(string) > max_length:
        return f'{string[:max_length // 2]}......{string[-max_length // 2:]}'
    return string


def process_model_prediction(item: Any):
    if isinstance(item, dict):
        return dict_to_markdown(item)
    elif isinstance(item, list):
        return '\n'.join([process_model_prediction(item) for item in item])
    else:
        return process_string(str(item))


def get_model_prediction(work_dir: str, model_name: str, dataset_name: str, subset_name: str):
    data_path = os.path.join(work_dir, OutputsStructure.REVIEWS_DIR, model_name)
    origin_df = pd.read_json(os.path.join(data_path, f'{dataset_name}_{subset_name}.jsonl'), lines=True)
    ds = []
    for i, item in origin_df.iterrows():
        raw_input = item['raw_input']
        raw_pred_answer = item['choices'][0]['message']['content']
        parsed_gold_answer = item['choices'][0]['review']['gold']
        parsed_pred_answer = item['choices'][0]['review']['pred']
        score = item['choices'][0]['review']['result']
        d = {
            'Input': process_model_prediction(raw_input),
            'Generated': process_model_prediction(raw_pred_answer),
            'Gold':
            '*Same as Input*' if parsed_gold_answer == raw_input else process_model_prediction(parsed_gold_answer),
            'Pred': process_model_prediction(parsed_pred_answer),
            'Score': process_model_prediction(score)
        }
        ds.append(d)

    df_subset = pd.DataFrame(ds)
    return df_subset


def get_table_data(data_review_df: pd.DataFrame, page: int = 1, rows_per_page: int = 1) -> pd.DataFrame:
    if data_review_df is None:
        return None
    logger.debug(f'page: {page}, rows_per_page: {rows_per_page}')
    start = (page - 1) * rows_per_page
    end = start + rows_per_page
    df_subset = data_review_df.iloc[start:end]
    return df_subset


@dataclass
class SidebarComponents:
    root_path: gr.Textbox
    reports_dropdown: gr.Dropdown
    load_btn: gr.Button


def create_sidebar():
    with gr.Column(scale=1):
        gr.Markdown('### ⚙️ Settings')
        root_path = gr.Textbox(label='Report(s) Root Path', value='./outputs', placeholder='./outputs', lines=1)
        reports_dropdown = gr.Dropdown(label='Select Report(s)', choices=[], multiselect=True, interactive=True)
        load_btn = gr.Button('Load & View')

    @reports_dropdown.focus(inputs=[root_path], outputs=[reports_dropdown])
    def update_dropdown_choices(root_path):
        folders = scan_for_report_folders(root_path)
        if len(folders) == 0:
            gr.Warning('No reports found, please check the path', duration=3)
        return gr.update(choices=folders)

    return SidebarComponents(root_path=root_path, reports_dropdown=reports_dropdown, load_btn=load_btn)


@dataclass
class SingleModelComponents:
    report_name: gr.Dropdown


def create_single_model_tab(sidebar: SidebarComponents):
    report_name = gr.Dropdown(label='Select Report', choices=[], interactive=True)
    work_dir = gr.State(None)
    model_name = gr.State(None)

    with gr.Accordion('Task Config', open=False):
        task_config = gr.JSON(value=None)

    report_list = gr.State([])
    report_df = gr.State(None)

    with gr.Tab('Datasets Overview'):
        gr.Markdown('### Dataset Components')
        sunburst_plot = gr.Plot(value=None, scale=1, label='Components')
        gr.Markdown('### Dataset Scores')
        score_plot = gr.Plot(value=None, scale=1, label='Scores')
        gr.Markdown('### Dataset Scores Table')
        score_table = gr.DataFrame(value=None)

    with gr.Tab('Dataset Details'):
        dataset_radio = gr.Radio(label='Select Dataset', choices=[], show_label=True, interactive=True)
        gr.Markdown('### Dataset Scores')
        dataset_plot = gr.Plot(value=None, scale=1, label='Scores')
        gr.Markdown('### Dataset Scores Table')
        dataset_table = gr.DataFrame(value=None)

        gr.Markdown('### Model Prediction')
        subset_radio = gr.Radio(label='Select Subset', choices=[], show_label=True, interactive=True)
        page_number = gr.Number(value=1, label='Page')
        data_review_df = gr.State(None)
        data_review_table = gr.DataFrame(
            value=None,
            datatype=['markdown', 'markdown', 'markdown', 'markdown', 'markdown'],
            column_widths=['35%', '35%', '10%', '10%', '10%'],
            wrap=True)

    @report_name.change(
        inputs=[sidebar.root_path, report_name],
        outputs=[report_list, report_df, task_config, dataset_radio, work_dir, model_name])
    def update_single_report_data(root_path, report_name):
        report_list, report_df, task_cfg = load_single_report(root_path, report_name)
        datasets = [report.dataset_name for report in report_list]
        work_dir = os.path.join(root_path, report_name.split('@')[0])
        model_name = report_name.split('@')[1].split(':')[0]
        return (report_list, report_df, task_cfg, gr.update(choices=datasets, value=datasets[0]), work_dir, model_name)

    @report_list.change(inputs=[report_list, report_df], outputs=[score_plot, score_table, sunburst_plot])
    def update_single_report_score(report_list, report_df):
        report_score_df = get_single_report_data(report_list)
        report_score_plot = plot_single_report_scores(report_score_df)
        report_sunburst_plot = plot_single_report_sunburst(report_df)
        return report_score_plot, report_score_df, report_sunburst_plot

    @dataset_radio.change(inputs=[dataset_radio, report_df], outputs=[dataset_plot, dataset_table, subset_radio])
    def update_single_report_dataset(dataset_name, report_df):
        logger.debug(f'Updating single report dataset: {dataset_name}')
        data_score_df = get_single_dataset_data(report_df, dataset_name)
        data_score_plot = plot_single_dataset_scores(data_score_df)
        subsets = data_score_df[ReportKey.subset_name].unique().tolist()
        logger.debug(f'subsets: {subsets}')
        return data_score_plot, data_score_df, gr.update(choices=subsets, value=subsets[0])

    @subset_radio.change(
        inputs=[work_dir, model_name, dataset_radio, subset_radio], outputs=[data_review_df, data_review_table])
    def update_single_report_subset(work_dir, model_name, dataset_name, subset_name):
        if not subset_name:
            return gr.skip()
        data_review_df = get_model_prediction(work_dir, model_name, dataset_name, subset_name)
        subset_df = get_table_data(data_review_df, 1)
        return data_review_df, subset_df

    @page_number.change(inputs=[data_review_df, page_number], outputs=[data_review_table])
    def update_table(data_review_df, page_number):
        return get_table_data(data_review_df, page_number)

    return SingleModelComponents(report_name=report_name)


def create_multi_model_tab():
    gr.Markdown('### 📈 Charts')
    multi_report_name = gr.Dropdown(label='Select Reports', choices=[], multiselect=True, interactive=True)
    main_plot = gr.Plot(value=None)
    reward_dist = gr.Plot(value=None)
    return multi_report_name, main_plot, reward_dist


with gr.Blocks(title='Evalscope Dashboard') as demo:
    gr.Markdown('# Evalscope Dashboard')

    with gr.Row():
        sidebar = create_sidebar()

        with gr.Column(scale=5) as main_content:
            initial_message = gr.Markdown('### Note: Select report(s) and click `Load & View` to view the data!')

            with gr.Column(visible=True) as content_group:
                with gr.Tabs():
                    with gr.Tab('Single Model'):
                        single = create_single_model_tab(sidebar)

                    with gr.Tab('Multi Model'):
                        multi = create_multi_model_tab()

    @sidebar.load_btn.click(
        inputs=[sidebar.reports_dropdown], outputs=[initial_message, content_group, single.report_name, multi[0]])
    def update_displays(reports_dropdown):
        if not reports_dropdown:
            gr.Warning('No reports found, please check the path', duration=3)
            return gr.skip()

        return (
            gr.update(visible=False),  # hide initial message
            gr.update(visible=True),  # show content area
            gr.update(choices=reports_dropdown, value=reports_dropdown[0]),  # update single model dropdown
            gr.update(choices=reports_dropdown, value=reports_dropdown)  # update multi model dropdown
        )


if __name__ == '__main__':
    demo.launch()
