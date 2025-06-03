import argparse
import glob
import gradio as gr
import numpy as np
import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import re
from dataclasses import dataclass
from typing import Any, List, Union

from evalscope.constants import DataCollection
from evalscope.report import Report, ReportKey, get_data_frame, get_report_list
from evalscope.utils.io_utils import OutputsStructure, yaml_to_dict
from evalscope.utils.logger import configure_logging, get_logger
from evalscope.version import __version__
from .arguments import add_argument
from .constants import DATASET_TOKEN, LATEX_DELIMITERS, MODEL_TOKEN, PLOTLY_THEME, REPORT_TOKEN

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


def plot_single_report_scores(df: pd.DataFrame):
    if df is None:
        return None
    logger.debug(f'df: {df}')
    plot = px.bar(df, x=df[ReportKey.dataset_name], y=df[ReportKey.score], text=df[ReportKey.score])

    width = 0.2 if len(df[ReportKey.dataset_name]) <= 5 else None
    plot.update_traces(width=width, texttemplate='%{text:.2f}', textposition='outside')
    plot.update_layout(uniformtext_minsize=12, uniformtext_mode='hide', yaxis=dict(range=[0, 1]), template=PLOTLY_THEME)
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
        template=PLOTLY_THEME,
        maxdepth=4)
    plot.update_traces(insidetextorientation='radial')
    plot.update_layout(margin=dict(t=10, l=10, r=10, b=10), coloraxis=dict(cmin=0, cmax=1), height=600)
    return plot


def get_single_dataset_df(df: pd.DataFrame, dataset_name: str):
    df = df[df[ReportKey.dataset_name] == dataset_name]
    styler = style_df(df, columns=[ReportKey.score])
    return df, styler


def get_report_analysis(report_list: List[Report], dataset_name: str) -> str:
    for report in report_list:
        if report.dataset_name == dataset_name:
            return report.analysis
    return 'N/A'


def plot_single_dataset_scores(df: pd.DataFrame):
    # TODO: add metric radio and relace category name
    plot = px.bar(
        df,
        x=df[ReportKey.metric_name],
        y=df[ReportKey.score],
        color=df[ReportKey.subset_name],
        text=df[ReportKey.score],
        barmode='group')

    width = 0.2 if len(df[ReportKey.subset_name]) <= 3 else None
    plot.update_traces(width=width, texttemplate='%{text:.2f}', textposition='outside')
    plot.update_layout(uniformtext_minsize=12, uniformtext_mode='hide', yaxis=dict(range=[0, 1]), template=PLOTLY_THEME)
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
        template=PLOTLY_THEME,
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        margin=dict(t=20, l=20, r=20, b=20))
    return fig


def convert_markdown_image(text):
    if not os.path.isfile(text):
        return text
    # Convert the image path to a markdown image tag
    if text.endswith('.png') or text.endswith('.jpg') or text.endswith('.jpeg'):
        text = os.path.abspath(text)
        image_tag = f'![image](gradio_api/file={text})'
        logger.debug(f'Converting image path to markdown: {text} -> {image_tag}')
        return image_tag
    return text


def convert_html_tags(text):
    # match begin label
    text = re.sub(r'<(\w+)>', r'[\1]', text)
    # match end label
    text = re.sub(r'</(\w+)>', r'[/\1]', text)
    return text


def process_string(string: str, max_length: int = 2048) -> str:
    string = convert_html_tags(string)  # for display labels e.g.
    if max_length and len(string) > max_length:
        return f'{string[:max_length // 2]}......{string[-max_length // 2:]}'
    return string


def dict_to_markdown(data) -> str:
    markdown_lines = []

    for key, value in data.items():
        bold_key = f'**{key}**'

        if isinstance(value, list):
            value_str = '\n' + '\n'.join([f'- {process_model_prediction(item, max_length=None)}' for item in value])
        elif isinstance(value, dict):
            value_str = dict_to_markdown(value)
        else:
            value_str = str(value)

        value_str = process_string(value_str, max_length=None)  # Convert HTML tags but don't truncate
        markdown_line = f'{bold_key}:\n{value_str}'
        markdown_lines.append(markdown_line)

    return '\n\n'.join(markdown_lines)


def process_model_prediction(item: Any, max_length: int = 2048) -> str:
    """
    Process model prediction output into a formatted string.

    Args:
        item: The item to process. Can be a string, list, or dictionary.
        max_length: The maximum length of the output string.

    Returns:
        A formatted string representation of the input.
    """
    if isinstance(item, dict):
        result = dict_to_markdown(item)
    elif isinstance(item, list):
        result = '\n'.join([f'- {process_model_prediction(i, max_length=None)}' for i in item])
    else:
        result = str(item)

    # Apply HTML tag conversion and truncation only at the final output
    if max_length is not None:
        return process_string(result, max_length)
    return result


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


def get_model_prediction(work_dir: str, model_name: str, dataset_name: str, subset_name: str):
    data_path = os.path.join(work_dir, OutputsStructure.REVIEWS_DIR, model_name)
    subset_name = subset_name.replace('/', '_')  # for collection report
    review_path = os.path.join(data_path, f'{dataset_name}_{subset_name}.jsonl')
    logger.debug(f'review_path: {review_path}')
    origin_df = pd.read_json(review_path, lines=True)

    ds = []
    for i, item in origin_df.iterrows():
        raw_input = item['raw_input']
        for choice in item['choices']:
            raw_pred_answer = choice['message']['content']
            parsed_gold_answer = choice['review']['gold']
            parsed_pred_answer = choice['review']['pred']
            score = choice['review']['result']
            raw_d = {
                'Input': raw_input,
                'Generated': raw_pred_answer,
                'Gold': parsed_gold_answer if parsed_gold_answer != raw_input else '*Same as Input*',
                'Pred': parsed_pred_answer,
                'Score': score,
                'NScore': normalize_score(score)
            }
            ds.append(raw_d)

    df_subset = pd.DataFrame(ds)
    return df_subset


@dataclass
class SidebarComponents:
    root_path: gr.Textbox
    reports_dropdown: gr.Dropdown
    load_btn: gr.Button


def create_sidebar(outputs_dir: str, lang: str):
    locale_dict = {
        'settings': {
            'zh': '设置',
            'en': 'Settings'
        },
        'report_root_path': {
            'zh': '报告根路径',
            'en': 'Report Root Path'
        },
        'select_reports': {
            'zh': '请选择报告',
            'en': 'Select Reports'
        },
        'load_btn': {
            'zh': '加载并查看',
            'en': 'Load & View'
        },
        'note': {
            'zh': '请选择报告并点击`加载并查看`来查看数据',
            'en': 'Please select reports and click `Load & View` to view the data'
        },
        'warning': {
            'zh': '没有找到报告，请检查路径',
            'en': 'No reports found, please check the path'
        }
    }

    gr.Markdown(f'## {locale_dict["settings"][lang]}')
    root_path = gr.Textbox(
        label=locale_dict['report_root_path'][lang], value=outputs_dir, placeholder=outputs_dir, lines=1)
    reports_dropdown = gr.Dropdown(
        label=locale_dict['select_reports'][lang], choices=[], multiselect=True, interactive=True)
    load_btn = gr.Button(locale_dict['load_btn'][lang])
    gr.Markdown(f'### {locale_dict["note"][lang]}')

    @reports_dropdown.focus(inputs=[root_path], outputs=[reports_dropdown])
    def update_dropdown_choices(root_path):
        folders = scan_for_report_folders(root_path)
        if len(folders) == 0:
            gr.Warning(locale_dict['warning'][lang], duration=3)
        return gr.update(choices=folders)

    return SidebarComponents(
        root_path=root_path,
        reports_dropdown=reports_dropdown,
        load_btn=load_btn,
    )


@dataclass
class VisualizationComponents:
    single_model: gr.Tab
    multi_model: gr.Tab


def create_visualization(sidebar: SidebarComponents, lang: str):
    locale_dict = {
        'visualization': {
            'zh': '可视化',
            'en': 'Visualization'
        },
        'single_model': {
            'zh': '单模型',
            'en': 'Single Model'
        },
        'multi_model': {
            'zh': '多模型',
            'en': 'Multi Model'
        }
    }
    with gr.Column(visible=True):
        gr.Markdown(f'## {locale_dict["visualization"][lang]}')
        with gr.Tabs():
            with gr.Tab(locale_dict['single_model'][lang]):
                single = create_single_model_tab(sidebar, lang)

            with gr.Tab(locale_dict['multi_model'][lang]):
                multi = create_multi_model_tab(sidebar, lang)
    return VisualizationComponents(
        single_model=single,
        multi_model=multi,
    )


@dataclass
class SingleModelComponents:
    report_name: gr.Dropdown


def create_single_model_tab(sidebar: SidebarComponents, lang: str):
    locale_dict = {
        'select_report': {
            'zh': '选择报告',
            'en': 'Select Report'
        },
        'task_config': {
            'zh': '任务配置',
            'en': 'Task Config'
        },
        'datasets_overview': {
            'zh': '数据集概览',
            'en': 'Datasets Overview'
        },
        'dataset_components': {
            'zh': '数据集组成',
            'en': 'Dataset Components'
        },
        'dataset_scores': {
            'zh': '数据集分数',
            'en': 'Dataset Scores'
        },
        'report_analysis': {
            'zh': '报告智能分析',
            'en': 'Report Intelligent Analysis'
        },
        'dataset_scores_table': {
            'zh': '数据集分数表',
            'en': 'Dataset Scores Table'
        },
        'dataset_details': {
            'zh': '数据集详情',
            'en': 'Dataset Details'
        },
        'select_dataset': {
            'zh': '选择数据集',
            'en': 'Select Dataset'
        },
        'model_prediction': {
            'zh': '模型预测',
            'en': 'Model Prediction'
        },
        'select_subset': {
            'zh': '选择子集',
            'en': 'Select Subset'
        },
        'answer_mode': {
            'zh': '答案模式',
            'en': 'Answer Mode'
        },
        'page': {
            'zh': '页码',
            'en': 'Page'
        },
        'score_threshold': {
            'zh': '分数阈值',
            'en': 'Score Threshold'
        },
    }

    # Update the UI components with localized labels
    report_name = gr.Dropdown(label=locale_dict['select_report'][lang], choices=[], interactive=True)
    work_dir = gr.State(None)
    model_name = gr.State(None)

    with gr.Accordion(locale_dict['task_config'][lang], open=False):
        task_config = gr.JSON(value=None)

    report_list = gr.State([])

    with gr.Tab(locale_dict['datasets_overview'][lang]):
        gr.Markdown(f'### {locale_dict["dataset_components"][lang]}')
        sunburst_plot = gr.Plot(value=None, scale=1, label=locale_dict['dataset_components'][lang])
        gr.Markdown(f'### {locale_dict["dataset_scores"][lang]}')
        score_plot = gr.Plot(value=None, scale=1, label=locale_dict['dataset_scores'][lang])
        gr.Markdown(f'### {locale_dict["dataset_scores_table"][lang]}')
        score_table = gr.DataFrame(value=None)

    with gr.Tab(locale_dict['dataset_details'][lang]):
        dataset_radio = gr.Radio(
            label=locale_dict['select_dataset'][lang], choices=[], show_label=True, interactive=True)
        # show dataset details
        with gr.Accordion(locale_dict['report_analysis'][lang], open=True):
            report_analysis = gr.Markdown(value='N/A', show_copy_button=True)
        gr.Markdown(f'### {locale_dict["dataset_scores"][lang]}')
        dataset_plot = gr.Plot(value=None, scale=1, label=locale_dict['dataset_scores'][lang])
        gr.Markdown(f'### {locale_dict["dataset_scores_table"][lang]}')
        dataset_table = gr.DataFrame(value=None)

        gr.Markdown(f'### {locale_dict["model_prediction"][lang]}')
        subset_select = gr.Dropdown(
            label=locale_dict['select_subset'][lang], choices=[], show_label=True, interactive=True)

        with gr.Row():
            answer_mode_radio = gr.Radio(
                label=locale_dict['answer_mode'][lang], choices=['All', 'Pass', 'Fail'], value='All', interactive=True)
            score_threshold = gr.Number(value=0.99, label=locale_dict['score_threshold'][lang], interactive=True)

        data_review_df = gr.State(None)
        filtered_review_df = gr.State(None)

        # show statistics
        with gr.Row(variant='panel'):
            with gr.Column():
                gr.Markdown('### *Counts*')
                answer_mode_counts = gr.Markdown('')
            with gr.Column():
                page_number = gr.Number(
                    value=1, label=locale_dict['page'][lang], minimum=1, maximum=1, step=1, interactive=True)

        # show data review table
        with gr.Row(variant='panel'):
            with gr.Column():
                gr.Markdown('### *Score*')
                score_text = gr.Markdown(
                    '', elem_id='score_text', latex_delimiters=LATEX_DELIMITERS, show_copy_button=True)
            with gr.Column():
                gr.Markdown('### *Normalized Score*')
                nscore = gr.Markdown('', elem_id='score_text', latex_delimiters=LATEX_DELIMITERS)

        with gr.Row(variant='panel'):
            with gr.Column():
                gr.Markdown('### *Gold*')
                gold_text = gr.Markdown(
                    '', elem_id='gold_text', latex_delimiters=LATEX_DELIMITERS, show_copy_button=True)
            with gr.Column():
                gr.Markdown('### *Pred*')
                pred_text = gr.Markdown(
                    '', elem_id='pred_text', latex_delimiters=LATEX_DELIMITERS, show_copy_button=True)

        with gr.Row(variant='panel'):
            with gr.Column():
                gr.Markdown('### *Input*')
                input_text = gr.Markdown(
                    '', elem_id='input_text', latex_delimiters=LATEX_DELIMITERS, show_copy_button=True)
            with gr.Column():
                gr.Markdown('### *Generated*')
                generated_text = gr.Markdown(
                    '', elem_id='generated_text', latex_delimiters=LATEX_DELIMITERS, show_copy_button=True)

    @report_name.change(
        inputs=[sidebar.root_path, report_name],
        outputs=[report_list, task_config, dataset_radio, work_dir, model_name])
    def update_single_report_data(root_path, report_name):
        report_list, datasets, task_cfg = load_single_report(root_path, report_name)
        work_dir = os.path.join(root_path, report_name.split(REPORT_TOKEN)[0])
        model_name = report_name.split(REPORT_TOKEN)[1].split(MODEL_TOKEN)[0]
        return (report_list, task_cfg, gr.update(choices=datasets, value=datasets[0]), work_dir, model_name)

    @report_list.change(inputs=[report_list], outputs=[score_plot, score_table, sunburst_plot])
    def update_single_report_score(report_list):
        report_score_df, styler = get_acc_report_df(report_list)
        report_score_plot = plot_single_report_scores(report_score_df)
        report_sunburst_plot = plot_single_report_sunburst(report_list)
        return report_score_plot, styler, report_sunburst_plot

    @gr.on(
        triggers=[dataset_radio.change, report_list.change],
        inputs=[dataset_radio, report_list],
        outputs=[dataset_plot, dataset_table, subset_select, data_review_df, report_analysis])
    def update_single_report_dataset(dataset_name, report_list):
        logger.debug(f'Updating single report dataset: {dataset_name}')
        report_df = get_data_frame(report_list)
        analysis = get_report_analysis(report_list, dataset_name)
        data_score_df, styler = get_single_dataset_df(report_df, dataset_name)
        data_score_plot = plot_single_dataset_scores(data_score_df)
        subsets = data_score_df[ReportKey.subset_name].unique().tolist()
        logger.debug(f'subsets: {subsets}')
        return data_score_plot, styler, gr.update(choices=subsets, value=None), None, analysis

    @gr.on(
        triggers=[subset_select.change],
        inputs=[work_dir, model_name, dataset_radio, subset_select],
        outputs=[data_review_df, page_number])
    def update_single_report_subset(work_dir, model_name, dataset_name, subset_name):
        if not subset_name:
            return gr.skip()
        data_review_df = get_model_prediction(work_dir, model_name, dataset_name, subset_name)
        return data_review_df, 1

    @gr.on(
        triggers=[data_review_df.change, answer_mode_radio.change, score_threshold.change],
        inputs=[data_review_df, answer_mode_radio, score_threshold],
        outputs=[filtered_review_df, page_number, answer_mode_counts])
    def filter_data(data_review_df, answer_mode, score_threshold):
        if data_review_df is None:
            return None, gr.update(value=1, maximum=1), ''

        all_count = len(data_review_df)
        pass_df = data_review_df[data_review_df['NScore'] >= score_threshold]
        pass_count = len(pass_df)
        fail_count = all_count - pass_count

        counts_text = f'### All: {all_count} | Pass: {pass_count} | Fail: {fail_count}'

        if answer_mode == 'Pass':
            filtered_df = pass_df
        elif answer_mode == 'Fail':
            filtered_df = data_review_df[data_review_df['NScore'] < score_threshold]
        else:
            filtered_df = data_review_df

        max_page = max(1, len(filtered_df))

        return (filtered_df, gr.update(value=1, maximum=max_page), counts_text)

    @gr.on(
        triggers=[filtered_review_df.change, page_number.change],
        inputs=[filtered_review_df, page_number, score_threshold],
        outputs=[input_text, generated_text, gold_text, pred_text, score_text, nscore])
    def update_table_components(filtered_df, page_number, score_threshold):
        if filtered_df is None or len(filtered_df) == 0:
            return '', '', '', '', '', ''

        # Get single row data for the current page
        start = (page_number - 1)
        if start >= len(filtered_df):
            return '', '', '', '', '', ''

        row = filtered_df.iloc[start]

        # Process the data for display
        input_md = process_model_prediction(row['Input'])
        generated_md = process_model_prediction(row['Generated'])
        gold_md = process_model_prediction(row['Gold'])
        pred_md = convert_markdown_image(process_model_prediction(row['Pred']))
        score_md = process_model_prediction(row['Score'])
        nscore_val = float(row['NScore']) if not pd.isna(row['NScore']) else 0.0

        if nscore_val >= score_threshold:
            nscore_val = f'<div style="background-color:rgb(45,104, 62); padding:10px;">{nscore_val}</div>'
        else:
            nscore_val = f'<div style="background-color:rgb(151, 31, 44); padding:10px;">{nscore_val}</div>'

        return input_md, generated_md, gold_md, pred_md, score_md, nscore_val

    return SingleModelComponents(report_name=report_name)


@dataclass
class MultiModelComponents:
    multi_report_name: gr.Dropdown


def create_multi_model_tab(sidebar: SidebarComponents, lang: str):
    locale_dict = {
        'select_reports': {
            'zh': '请选择报告',
            'en': 'Select Reports'
        },
        'model_radar': {
            'zh': '模型对比雷达',
            'en': 'Model Comparison Radar'
        },
        'model_scores': {
            'zh': '模型对比分数',
            'en': 'Model Comparison Scores'
        }
    }
    multi_report_name = gr.Dropdown(
        label=locale_dict['select_reports'][lang], choices=[], multiselect=True, interactive=True)
    gr.Markdown(locale_dict['model_radar'][lang])
    radar_plot = gr.Plot(value=None)
    gr.Markdown(locale_dict['model_scores'][lang])
    score_table = gr.DataFrame(value=None)

    @multi_report_name.change(inputs=[sidebar.root_path, multi_report_name], outputs=[radar_plot, score_table])
    def update_multi_report_data(root_path, multi_report_name):
        if not multi_report_name:
            return gr.skip()
        report_list = load_multi_report(root_path, multi_report_name)
        report_df, _ = get_acc_report_df(report_list)
        report_radar_plot = plot_multi_report_radar(report_df)
        _, styler = get_compare_report_df(report_df)
        return report_radar_plot, styler

    return MultiModelComponents(multi_report_name=multi_report_name)


def create_app(args: argparse.Namespace):
    configure_logging(debug=args.debug)
    lang = args.lang

    locale_dict = {
        'title': {
            'zh': '📈 EvalScope 看板',
            'en': '📈 Evalscope Dashboard'
        },
        'star_beggar': {
            'zh':
            '喜欢<a href=\"https://github.com/modelscope/evalscope\" target=\"_blank\">EvalScope</a>就动动手指给我们加个star吧 🥺 ',
            'en':
            'If you like <a href=\"https://github.com/modelscope/evalscope\" target=\"_blank\">EvalScope</a>, '
            'please take a few seconds to star us 🥺 '
        },
        'note': {
            'zh': '请选择报告',
            'en': 'Please select reports'
        }
    }

    with gr.Blocks(title='Evalscope Dashboard') as demo:
        gr.HTML(f'<h1 style="text-align: left;">{locale_dict["title"][lang]} (v{__version__})</h1>')
        with gr.Row():
            with gr.Column(scale=0, min_width=35):
                toggle_btn = gr.Button('<')
            with gr.Column(scale=1):
                gr.HTML(f'<h3 style="text-align: left;">{locale_dict["star_beggar"][lang]}</h3>')

        with gr.Row():
            with gr.Column(scale=1) as sidebar_column:
                sidebar_visible = gr.State(True)
                sidebar = create_sidebar(args.outputs, lang)

            with gr.Column(scale=5):
                visualization = create_visualization(sidebar, lang)

        @sidebar.load_btn.click(
            inputs=[sidebar.reports_dropdown],
            outputs=[visualization.single_model.report_name, visualization.multi_model.multi_report_name])
        def update_displays(reports_dropdown):
            if not reports_dropdown:
                gr.Warning(locale_dict['note'][lang], duration=3)
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

    demo.launch(
        share=args.share,
        server_name=args.server_name,
        server_port=args.server_port,
        debug=args.debug,
        allowed_paths=args.allowed_paths,
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    add_argument(parser)
    args = parser.parse_args()
    create_app(args)
