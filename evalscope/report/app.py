import argparse
import glob
import gradio as gr
import numpy as np
import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dataclasses import dataclass
from typing import Any, List, Union

from evalscope.constants import DataCollection
from evalscope.report import Report, ReportKey, get_data_frame, get_report_list
from evalscope.utils.io_utils import OutputsStructure, yaml_to_dict
from evalscope.utils.logger import configure_logging, get_logger

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
                datasets.append(os.path.basename(dataset_item).split('.')[0])
            datasets = ','.join(datasets)
            reports.append(f'{os.path.basename(folder)}@{os.path.basename(model_item)}:{datasets}')

    reports = sorted(reports, reverse=True)
    logger.debug(f'reports: {reports}')
    return reports


def process_report_name(report_name: str):
    prefix, report_name = report_name.split('@')
    model_name, datasets = report_name.split(':')
    datasets = datasets.split(',')
    return prefix, model_name, datasets


def load_single_report(root_path: str, report_name: str):
    prefix, model_name, datasets = process_report_name(report_name)
    report_path_list = os.path.join(root_path, prefix, OutputsStructure.REPORTS_DIR, model_name)
    report_list = get_report_list([report_path_list])

    task_cfg_path = glob.glob(os.path.join(root_path, prefix, OutputsStructure.CONFIGS_DIR, '*.yaml'))[0]
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
    return df


def get_compare_report_df(acc_df: pd.DataFrame):
    df = acc_df.pivot_table(index=ReportKey.model_name, columns=ReportKey.dataset_name, values=ReportKey.score)
    df.reset_index(inplace=True)
    styler = df.style.background_gradient(cmap='RdYlGn', vmin=0.0, vmax=1.0, axis=0)
    styler.format(precision=4)
    return styler


def plot_single_report_scores(df: pd.DataFrame):
    plot = px.bar(
        df,
        x=df[ReportKey.dataset_name],
        y=df[ReportKey.score],
        color=df[ReportKey.dataset_name],
        template='plotly_dark')
    return plot


def plot_single_report_sunburst(report_list: List[Report]):
    if report_list[0].name == DataCollection.NAME:
        df = get_data_frame(report_list)
        categories = sorted([i for i in df.columns if i.startswith(ReportKey.category_prefix)])
        path = categories + [ReportKey.subset_name]
    else:
        df = get_data_frame(report_list, flatten_metrics=False)
        categories = sorted([i for i in df.columns if i.startswith(ReportKey.category_prefix)])
        path = [ReportKey.dataset_name] + categories + [ReportKey.subset_name]
    logger.debug(f'df: {df}')
    df[categories] = df[categories].fillna('default')  # NOTE: fillna for empty categories
    plot = px.sunburst(
        df,
        path=path,
        values=ReportKey.num,
        color=ReportKey.score,
        color_continuous_scale='RdYlGn',  # see https://plotly.com/python/builtin-colorscales/
        color_continuous_midpoint=np.average(df[ReportKey.score], weights=df[ReportKey.num]),
        template='plotly_dark',
        maxdepth=3)
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


def plot_multi_report_radar(df: pd.DataFrame):
    fig = go.Figure()

    grouped = df.groupby(ReportKey.model_name)
    common_datasets = set.intersection(*[set(group[ReportKey.dataset_name]) for _, group in grouped])

    for model_name, group in grouped:
        common_group = group[group[ReportKey.dataset_name].isin(common_datasets)]
        fig.add_trace(
            go.Scatterpolar(
                r=common_group[ReportKey.score],
                theta=common_group[ReportKey.dataset_name],
                name=model_name,
                fill='toself'))

    fig.update_layout(
        template='plotly_dark',
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        margin=dict(t=20, l=20, r=20, b=20))
    return fig


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


def process_string(string: str, max_length: int = 2048) -> str:
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


def normalize_score(score):
    if isinstance(score, bool):
        return 1.0 if score else 0.0
    elif isinstance(score, dict):
        for key in score:
            return float(score[key])
        return 0.0
    else:
        try:
            return float(score)
        except (ValueError, TypeError):
            return 0.0


def get_model_prediction(work_dir: str, model_name: str, dataset_name: str, subset_name: str):
    data_path = os.path.join(work_dir, OutputsStructure.REVIEWS_DIR, model_name)
    subset_name = subset_name.replace('/', '_')  # for collection report
    origin_df = pd.read_json(os.path.join(data_path, f'{dataset_name}_{subset_name}.jsonl'), lines=True)
    ds = []
    for i, item in origin_df.iterrows():
        raw_input = item['raw_input']
        raw_pred_answer = item['choices'][0]['message']['content']
        parsed_gold_answer = item['choices'][0]['review']['gold']
        parsed_pred_answer = item['choices'][0]['review']['pred']
        score = item['choices'][0]['review']['result']
        raw_d = {
            'Input': raw_input,
            'Generated': raw_pred_answer,
            'Gold': parsed_gold_answer if parsed_gold_answer != raw_input else '*Same as Input*',
            'Pred': parsed_pred_answer if parsed_pred_answer != raw_pred_answer else '*Same as Generated*',
            'Score': score,
            'NScore': normalize_score(score)
        }
        ds.append(raw_d)

    df_subset = pd.DataFrame(ds)
    return df_subset


def get_table_data(data_review_df: pd.DataFrame, page: int = 1, rows_per_page: int = 1) -> pd.DataFrame:
    if data_review_df is None:
        return None

    logger.debug(f'page: {page}, rows_per_page: {rows_per_page}')
    start = (page - 1) * rows_per_page
    end = start + rows_per_page
    df_subset = data_review_df.iloc[start:end].copy()
    df_subset['Input'] = df_subset['Input'].map(process_model_prediction).astype(str)
    df_subset['Score'] = df_subset['Score'].map(process_model_prediction).astype(str)
    return df_subset


@dataclass
class SidebarComponents:
    root_path: gr.Textbox
    reports_dropdown: gr.Dropdown
    load_btn: gr.Button


def create_sidebar(outputs_dir: str):
    gr.Markdown('## Settings')
    root_path = gr.Textbox(label='Report(s) Root Path', value=outputs_dir, placeholder=outputs_dir, lines=1)
    reports_dropdown = gr.Dropdown(label='Select Report(s)', choices=[], multiselect=True, interactive=True)
    load_btn = gr.Button('Load & View')
    gr.Markdown('### Note: Select report(s) and click `Load & View` to view the data!')

    @reports_dropdown.focus(inputs=[root_path], outputs=[reports_dropdown])
    def update_dropdown_choices(root_path):
        folders = scan_for_report_folders(root_path)
        if len(folders) == 0:
            gr.Warning('No reports found, please check the path', duration=3)
        return gr.update(choices=folders)

    return SidebarComponents(
        root_path=root_path,
        reports_dropdown=reports_dropdown,
        load_btn=load_btn,
    )


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
        with gr.Row():
            answer_mode_radio = gr.Radio(
                label='Answer Mode', choices=['All', 'Pass', 'Fail'], value='All', interactive=True)
            page_number = gr.Number(value=1, label='Page', minimum=1, maximum=1, step=1, interactive=True)
        answer_mode_counts = gr.Markdown('', label='Counts')
        data_review_df = gr.State(None)
        filtered_review_df = gr.State(None)
        data_review_table = gr.DataFrame(
            value=None,
            datatype=['markdown', 'markdown', 'markdown', 'markdown', 'markdown', 'number'],
            # column_widths=['500px', '500px'],
            wrap=True,
            latex_delimiters=[{
                'left': '$$',
                'right': '$$',
                'display': True
            }, {
                'left': '$',
                'right': '$',
                'display': False
            }, {
                'left': '\\(',
                'right': '\\)',
                'display': False
            }, {
                'left': '\\[',
                'right': '\\]',
                'display': True
            }],
            max_height=500)

    @report_name.change(
        inputs=[sidebar.root_path, report_name],
        outputs=[report_list, task_config, dataset_radio, work_dir, model_name])
    def update_single_report_data(root_path, report_name):
        report_list, datasets, task_cfg = load_single_report(root_path, report_name)
        work_dir = os.path.join(root_path, report_name.split('@')[0])
        model_name = report_name.split('@')[1].split(':')[0]
        return (report_list, task_cfg, gr.update(choices=datasets, value=datasets[0]), work_dir, model_name)

    @report_list.change(inputs=[report_list], outputs=[score_plot, score_table, sunburst_plot])
    def update_single_report_score(report_list):
        report_score_df = get_acc_report_df(report_list)
        report_score_plot = plot_single_report_scores(report_score_df)
        report_sunburst_plot = plot_single_report_sunburst(report_list)
        return report_score_plot, report_score_df, report_sunburst_plot

    @gr.on(
        triggers=[dataset_radio.change, report_list.change],
        inputs=[dataset_radio, report_list],
        outputs=[dataset_plot, dataset_table, subset_radio])
    def update_single_report_dataset(dataset_name, report_list):
        logger.debug(f'Updating single report dataset: {dataset_name}')
        report_df = get_data_frame(report_list)
        data_score_df = get_single_dataset_data(report_df, dataset_name)
        data_score_plot = plot_single_dataset_scores(data_score_df)
        subsets = data_score_df[ReportKey.subset_name].unique().tolist()
        logger.debug(f'subsets: {subsets}')
        return data_score_plot, data_score_df, gr.update(choices=subsets, value=subsets[0])

    @subset_radio.change(
        inputs=[work_dir, model_name, dataset_radio, subset_radio], outputs=[data_review_df, page_number])
    def update_single_report_subset(work_dir, model_name, dataset_name, subset_name):
        if not subset_name:
            return gr.skip()
        data_review_df = get_model_prediction(work_dir, model_name, dataset_name, subset_name)
        return data_review_df, 1

    @gr.on(
        triggers=[data_review_df.change, answer_mode_radio.change],
        inputs=[data_review_df, answer_mode_radio],
        outputs=[filtered_review_df, page_number, answer_mode_counts])
    def filter_data(data_review_df, answer_mode):
        if data_review_df is None:
            return None, gr.update(value=1, maximum=1), ''

        all_count = len(data_review_df)
        pass_df = data_review_df[data_review_df['NScore'] >= 0.99]
        pass_count = len(pass_df)
        fail_count = all_count - pass_count

        counts_text = f'### All: {all_count} | Pass: {pass_count} | Fail: {fail_count}'

        if answer_mode == 'Pass':
            filtered_df = pass_df
        elif answer_mode == 'Fail':
            filtered_df = data_review_df[data_review_df['NScore'] < 0.99]
        else:
            filtered_df = data_review_df

        max_page = max(1, len(filtered_df))

        return (filtered_df, gr.update(value=1, maximum=max_page), counts_text)

    @gr.on(
        triggers=[filtered_review_df.change, page_number.change],
        inputs=[filtered_review_df, page_number],
        outputs=[data_review_table])
    def update_table(filtered_df, page_number):
        subset_df = get_table_data(filtered_df, page_number)
        if subset_df is None:
            return gr.skip()
        return subset_df

    return SingleModelComponents(report_name=report_name)


@dataclass
class MultiModelComponents:
    multi_report_name: gr.Dropdown


def create_multi_model_tab(sidebar: SidebarComponents):
    multi_report_name = gr.Dropdown(label='Select Reports', choices=[], multiselect=True, interactive=True)
    gr.Markdown('### Model Radar')
    radar_plot = gr.Plot(value=None)
    gr.Markdown('### Model Scores')
    score_table = gr.DataFrame(value=None)

    @multi_report_name.change(inputs=[sidebar.root_path, multi_report_name], outputs=[radar_plot, score_table])
    def update_multi_report_data(root_path, multi_report_name):
        if not multi_report_name:
            return gr.skip()
        report_list = load_multi_report(root_path, multi_report_name)
        report_df = get_acc_report_df(report_list)
        report_radar_plot = plot_multi_report_radar(report_df)
        report_compare_df = get_compare_report_df(report_df)
        return report_radar_plot, report_compare_df

    return MultiModelComponents(multi_report_name=multi_report_name)


def create_app(args: argparse.Namespace):
    configure_logging(debug=args.debug)

    with gr.Blocks(title='Evalscope Dashboard') as demo:
        with gr.Row():
            with gr.Column(scale=0, min_width=35):
                toggle_btn = gr.Button('<')
            with gr.Column(scale=1):
                gr.HTML('<h1 style="text-align: left;">Evalscope Dashboard</h1>')  # 文本列

        with gr.Row():
            with gr.Column(scale=1) as sidebar_column:
                sidebar_visible = gr.State(True)
                sidebar = create_sidebar(args.outputs)

            with gr.Column(scale=5):

                with gr.Column(visible=True):
                    gr.Markdown('## Visualization')
                    with gr.Tabs():
                        with gr.Tab('Single Model'):
                            single = create_single_model_tab(sidebar)

                        with gr.Tab('Multi Model'):
                            multi = create_multi_model_tab(sidebar)

        @sidebar.load_btn.click(
            inputs=[sidebar.reports_dropdown], outputs=[single.report_name, multi.multi_report_name])
        def update_displays(reports_dropdown):
            if not reports_dropdown:
                gr.Warning('No reports found, please check the path', duration=3)
                return gr.skip()

            return (
                gr.update(choices=reports_dropdown, value=reports_dropdown[0]),  # update single model dropdown
                gr.update(choices=reports_dropdown, value=reports_dropdown)  # update multi model dropdown
            )

        @toggle_btn.click(inputs=[sidebar_visible], outputs=[sidebar_column, sidebar_visible, toggle_btn])
        def toggle_sidebar(visible):
            new_visible = not visible
            text = '<' if new_visible else '>'
            return gr.update(visible=new_visible), new_visible, gr.update(value=text)

    demo.launch(share=args.share, server_name=args.server_name, server_port=args.server_port, debug=args.debug)


def add_argument(parser: argparse.ArgumentParser):
    parser.add_argument('--share', action='store_true', help='Share the app.')
    parser.add_argument('--server-name', type=str, default='0.0.0.0', help='The server name.')
    parser.add_argument('--server-port', type=int, default=None, help='The server port.')
    parser.add_argument('--debug', action='store_true', help='Debug the app.')
    parser.add_argument('--outputs', type=str, default='./outputs', help='The outputs dir.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    add_argument(parser)
    args = parser.parse_args()
    create_app(args)
