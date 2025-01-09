import glob
import gradio as gr
import numpy as np
import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import List, Union

from evalscope.report import Report, gen_data_frame
from evalscope.utils.io_utils import OutputsStructure, yaml_to_dict


# 生成模拟数据
def generate_sample_data(n_samples=100):
    # 时间序列数据
    x = np.linspace(0, 2000, n_samples)
    y1 = np.sin(x / 200) - 0.2 + np.random.normal(0, 0.05, n_samples)
    y2 = np.cos(x / 200) + 0.2 + np.random.normal(0, 0.05, n_samples)

    # 生成奖励数据
    rewards = np.random.normal(-0.5, 0.3, n_samples)

    return x, y1, y2, rewards


def create_main_plot():
    x, y1, y2, _ = generate_sample_data()
    fig = go.Figure()

    # 添加两条线
    fig.add_trace(go.Scatter(x=x, y=y1, name='ref_reward', line=dict(color='#00BFFF')))
    fig.add_trace(go.Scatter(x=x, y=y2, name='reward', line=dict(color='#FF69B4')))

    # 设置布局
    fig.update_layout(showlegend=True, margin=dict(l=0, r=0, t=30, b=0), height=300)

    return fig


def create_reward_distribution():
    _, _, _, rewards = generate_sample_data()
    fig = go.Figure()

    fig.add_trace(go.Histogram(x=rewards, nbinsx=30, name='Reward Distribution'))

    fig.update_layout(showlegend=False, margin=dict(l=0, r=0, t=30, b=0), height=200)

    return fig


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


def plot_single_dataset_scores(df: pd.DataFrame):
    plot = px.bar(df, x='Dataset Name', y='Score', color='Dataset Name', template='plotly_dark')
    plot.update_traces(showlegend=True, legendgroup=None, legendgrouptitle_text='')
    plot.update_layout(
        legend=dict(orientation='h', yanchor='bottom', y=-0.3, xanchor='center', x=0.5, title=None), margin=dict(b=100))
    return plot


with gr.Blocks(title='Evalscope Dashboard') as demo:
    gr.Markdown('# Evalscope Dashboard')
    with gr.Row():
        # Left Sidebar
        with gr.Column(scale=1) as sidebar:
            gr.Markdown('### ⚙️ Settings')
            root_path = gr.Textbox(label='Report(s) Root Path', value='./outputs', placeholder='./outputs', lines=1)
            reports_dropdown = gr.Dropdown(label='Select Report(s)', choices=[], multiselect=True, interactive=True)
            load_btn = gr.Button('Load & View')

        # Right Main Content Area
        with gr.Column(scale=5) as main_content:
            initial_message = gr.Markdown('### Note: Select report(s) and click `Load & View` to view the data!')

            with gr.Column(visible=True) as content_group:
                with gr.Tab('Single Model'):
                    single_report_name = gr.Dropdown(label='Select Report', choices=[], interactive=True)
                    with gr.Accordion('Task Config', open=False):
                        single_task_config = gr.JSON(value=None)

                    single_report_list = gr.State([])
                    single_report_df = gr.State(None)

                    with gr.Tab('Dataset Scores'):
                        single_plot = gr.Plot(value=None)
                        single_score_table = gr.DataFrame(value=None)

                    with gr.Tab('Data Review'):
                        single_datasets_radio = gr.Radio(
                            label='Select Dataset', choices=[], show_label=True, interactive=True)
                        data_review_table = gr.DataFrame(value=None)

                with gr.Tab('Multi Model'):
                    gr.Markdown('### 📈 Charts')

                    multi_report_name = gr.Dropdown(
                        label='Select Reports', choices=[], multiselect=True, interactive=True)
                    main_plot = gr.Plot(value=None)
                    reward_dist = gr.Plot(value=None)

    #  Update report dropdown choices
    @reports_dropdown.focus(inputs=[root_path], outputs=[reports_dropdown])
    def update_dropdown_choices(root_path):
        folders = scan_for_report_folders(root_path)
        if len(folders) == 0:
            gr.Warning('No reports found, please check the path', duration=3)
        return gr.update(choices=folders)

    # Load & View
    @load_btn.click(
        inputs=[reports_dropdown], outputs=[initial_message, content_group, single_report_name, multi_report_name])
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
    @single_report_name.change(
        inputs=[root_path, single_report_name],
        outputs=[single_report_list, single_report_df, single_task_config, single_datasets_radio])
    def update_single_report_data(root_path, single_report_name):
        report_list, data_frame, task_cfg = load_single_report(root_path, single_report_name)
        datasets = [report.dataset_name for report in report_list]
        return report_list, data_frame, task_cfg, gr.update(choices=datasets, value=datasets[0])

    # update single report score and plot
    @single_report_list.change(inputs=[single_report_list], outputs=[single_plot, single_score_table])
    def update_single_report_score(single_report_list):
        df = get_single_report_data(single_report_list)
        plot = plot_single_dataset_scores(df)
        return plot, df


if __name__ == '__main__':
    demo.launch()
