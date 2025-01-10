import glob
import gradio as gr
import numpy as np
import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dataclasses import dataclass
from typing import List, Union

from evalscope.report import Report, gen_data_frame
from evalscope.utils.io_utils import OutputsStructure, yaml_to_dict


def scan_for_report_folders(root_path):
    """Scan for folders containing reports subdirectories"""
    print(root_path)
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
            reports.append(f"{os.path.basename(folder)}@{os.path.basename(model_item)}:{datasets}")

    return sorted(reports, reverse=True)


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
            'Model Name': report.model_name,
            'Dataset Name': report.dataset_name,
            'Score': report.score,
            'Num Samples': report.metrics[0].num,
        })
    df = pd.DataFrame.from_dict(data_dict, orient='columns')
    return df


def plot_single_report_scores(df: pd.DataFrame):
    plot = px.bar(df, x='Dataset Name', y='Score', color='Dataset Name', template='plotly_dark')
    return plot


def get_single_dataset_data(df: pd.DataFrame, dataset_name: str):
    return df[df['Dataset Name'] == dataset_name]


def plot_single_dataset_scores(df: pd.DataFrame):
    if df['Metric Name'].nunique() > 1:
        plot = px.bar(df, x='Metric Name', y='Score', color='Subset Name', template='plotly_dark')
    else:
        plot = px.bar(df, x='Subset Name', y='Score', color='Subset Name', template='plotly_dark')
    return plot


def get_model_prediction(work_dir: str, model_name: str, dataset_name: str, subset_name: str):
    data_path = os.path.join(work_dir, OutputsStructure.REVIEWS_DIR, model_name)
    df = pd.read_json(os.path.join(data_path, f'{dataset_name}_{subset_name}.jsonl'), lines=True)
    return df


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
    return SidebarComponents(root_path=root_path, reports_dropdown=reports_dropdown, load_btn=load_btn)


@dataclass
class SingleModelComponents:
    report_name: gr.Dropdown
    work_dir: gr.State
    model_name: gr.State
    task_config: gr.JSON
    report_list: gr.State
    report_df: gr.State
    score_plot: gr.Plot
    score_table: gr.DataFrame
    dataset_radio: gr.Radio
    dataset_plot: gr.Plot
    dataset_table: gr.DataFrame
    data_review_table: gr.DataFrame
    subset_radio: gr.Radio


def create_single_model_tab():
    report_name = gr.Dropdown(label='Select Report', choices=[], interactive=True)
    work_dir = gr.State(None)
    model_name = gr.State(None)

    with gr.Accordion('Task Config', open=False):
        task_config = gr.JSON(value=None)

    report_list = gr.State([])
    report_df = gr.State(None)

    with gr.Tab('Dataset Scores'):
        score_plot = gr.Plot(value=None)
        score_table = gr.DataFrame(value=None)

    with gr.Tab('Data Review'):
        dataset_radio = gr.Radio(label='Select Dataset', choices=[], show_label=True, interactive=True)
        dataset_plot = gr.Plot(value=None)
        dataset_table = gr.DataFrame(value=None)

        gr.Markdown('### Model Prediction')
        subset_radio = gr.Radio(label='Select Subset', choices=[], show_label=True, interactive=True)
        data_review_table = gr.DataFrame(value=None)

    return SingleModelComponents(
        report_name=report_name,
        work_dir=work_dir,
        model_name=model_name,
        task_config=task_config,
        report_list=report_list,
        report_df=report_df,
        score_plot=score_plot,
        score_table=score_table,
        dataset_radio=dataset_radio,
        dataset_plot=dataset_plot,
        dataset_table=dataset_table,
        subset_radio=subset_radio,
        data_review_table=data_review_table)


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
                        single_model = create_single_model_tab()

                    with gr.Tab('Multi Model'):
                        multi_model_components = create_multi_model_tab()

    #  Update report dropdown choices
    @sidebar.reports_dropdown.focus(inputs=[sidebar.root_path], outputs=[sidebar.reports_dropdown])
    def update_dropdown_choices(root_path):
        folders = scan_for_report_folders(root_path)
        if len(folders) == 0:
            gr.Warning('No reports found, please check the path', duration=3)
        return gr.update(choices=folders)

    # Load & View
    @sidebar.load_btn.click(
        inputs=[sidebar.reports_dropdown],
        outputs=[initial_message, content_group, single_model.report_name, multi_model_components[0]])
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

    # load single report data
    @single_model.report_name.change(
        inputs=[sidebar.root_path, single_model.report_name],
        outputs=[
            single_model.report_list, single_model.report_df, single_model.task_config, single_model.dataset_radio,
            single_model.work_dir, single_model.model_name
        ])
    def update_single_report_data(root_path, report_name):
        report_list, data_frame, task_cfg = load_single_report(root_path, report_name)
        datasets = [report.dataset_name for report in report_list]
        work_dir = os.path.join(root_path, report_name.split('@')[0])
        model_name = report_name.split('@')[1].split(':')[0]
        return (report_list, data_frame, task_cfg, gr.update(choices=datasets, value=datasets[0]), work_dir, model_name)

    # update single report score and plot
    @single_model.report_list.change(
        inputs=[single_model.report_list], outputs=[single_model.score_plot, single_model.score_table])
    def update_single_report_score(report_list):
        report_score_df = get_single_report_data(report_list)
        report_score_plot = plot_single_report_scores(report_score_df)
        return report_score_plot, report_score_df

    # update single report dataset score and plot
    @single_model.dataset_radio.change(
        inputs=[single_model.dataset_radio, single_model.report_df],
        outputs=[single_model.dataset_plot, single_model.dataset_table, single_model.subset_radio])
    def update_single_report_dataset(dataset_name, report_df):
        print(dataset_name)
        data_score_df = get_single_dataset_data(report_df, dataset_name)
        data_score_plot = plot_single_dataset_scores(data_score_df)
        subsets = data_score_df['Subset Name'].unique().tolist()
        print(f'subsets: {subsets}')
        return data_score_plot, data_score_df, gr.update(choices=subsets, value=subsets[0])

    @single_model.subset_radio.change(
        inputs=[single_model.work_dir, single_model.model_name, single_model.dataset_radio, single_model.subset_radio],
        outputs=[single_model.data_review_table])
    def update_single_report_subset(work_dir, model_name, dataset_name, subset_name):
        data_review_df = get_model_prediction(work_dir, model_name, dataset_name, subset_name)
        return data_review_df


if __name__ == '__main__':
    demo.launch()
