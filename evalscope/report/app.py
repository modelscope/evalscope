import glob
import gradio as gr
import numpy as np
import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from evalscope.utils.io_utils import OutputsStructure


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


def create_sample_table():
    data = {
        'Index': range(5),
        'Prompt': [f'这是示例提示 {i}' for i in range(5)],
        'Response': [f'这是示例回复 {i}' for i in range(5)],
        'Reward': np.random.normal(0, 0.1, 5)
    }
    df = pd.DataFrame(data)
    return df


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
        for report_item in glob.glob(os.path.join(reports_path, '*')):
            if os.path.isdir(report_item):
                reports.append(f"{os.path.basename(folder)}/{os.path.basename(report_item)}")

    return sorted(reports, reverse=True)


def load_report(report_path):
    pass


with gr.Blocks(title='Evalscope Dashboard') as demo:
    gr.Markdown('# Evalscope Dashboard')
    with gr.Row():
        # Left Sidebar
        with gr.Column(scale=1) as sidebar:
            gr.Markdown('### ⚙️ Settings')
            folder_path = gr.Textbox(label='Report(s) Root Path', value='./outputs', placeholder='./outputs', lines=1)
            reports_dropdown = gr.Dropdown(label='Select Report(s)', choices=[], multiselect=True, interactive=True)
            load_btn = gr.Button('Load & View')

            with gr.Accordion('More Settings', open=False):
                pass

        # Right Main Content Area
        with gr.Column(scale=4) as main_content:
            initial_message = gr.Markdown('### Note: Select report(s) and click `Load & View` to view the data!')

            with gr.Column(visible=True) as content_group:
                with gr.Tab('Single Model'):
                    single_report_name = gr.Dropdown(label='Select Report', choices=[], interactive=True)

                    gr.Markdown('### 🔍 Data Review')
                    data_review_table = gr.DataFrame(
                        value=None,
                        headers=['Index', 'Prompt', 'Response', 'Reward'],
                    )

                with gr.Tab('Multi Model'):
                    gr.Markdown('### 📈 Charts')

                    multi_report_name = gr.Dropdown(
                        label='Select Reports', choices=[], multiselect=True, interactive=True)
                    main_plot = gr.Plot(value=None)
                    reward_dist = gr.Plot(value=None)

    @reports_dropdown.focus(inputs=[folder_path], outputs=[reports_dropdown])
    def update_dropdown_choices(path):
        folders = scan_for_report_folders(path)
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

        print(reports_dropdown)
        return (
            gr.update(visible=False),  # hide initial message
            gr.update(visible=True),  # show content area
            gr.update(choices=reports_dropdown, value=reports_dropdown[0]),  # update single model dropdown
            gr.update(choices=reports_dropdown, value=reports_dropdown)  # update multi model dropdown
        )

    @single_report_name.change(inputs=[single_report_name], outputs=[data_review_table])
    def update_single_report_data(single_report_name):
        print(single_report_name)
        pass


if __name__ == '__main__':
    demo.launch()
