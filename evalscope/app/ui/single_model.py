"""
Single model components for the Evalscope dashboard.
"""
import gradio as gr
import os
import pandas as pd
from dataclasses import dataclass
from typing import TYPE_CHECKING

from evalscope.report import Report, ReportKey, get_data_frame
from evalscope.utils.logger import get_logger
from ..constants import DATASET_TOKEN, LATEX_DELIMITERS, MODEL_TOKEN, REPORT_TOKEN
from ..utils.data_utils import (
    get_acc_report_df,
    get_model_prediction,
    get_report_analysis,
    get_single_dataset_df,
    load_single_report,
)
from ..utils.localization import get_single_model_locale
from ..utils.text_utils import convert_markdown_image, process_json_content, process_model_prediction
from ..utils.visualization import plot_single_dataset_scores, plot_single_report_scores, plot_single_report_sunburst

if TYPE_CHECKING:
    from .sidebar import SidebarComponents

logger = get_logger()


@dataclass
class SingleModelComponents:
    report_name: gr.Dropdown


def create_single_model_tab(sidebar: 'SidebarComponents', lang: str):
    locale_dict = get_single_model_locale(lang)

    # Update the UI components with localized labels
    report_name = gr.Dropdown(label=locale_dict['select_report'], choices=[], interactive=True)
    work_dir = gr.State(None)
    model_name = gr.State(None)

    with gr.Accordion(locale_dict['task_config'], open=False):
        task_config = gr.JSON(value=None)

    report_list = gr.State([])

    with gr.Tab(locale_dict['datasets_overview']):
        gr.Markdown(f'### {locale_dict["dataset_components"]}')
        sunburst_plot = gr.Plot(value=None, scale=1, label=locale_dict['dataset_components'])
        gr.Markdown(f'### {locale_dict["dataset_scores"]}')
        score_plot = gr.Plot(value=None, scale=1, label=locale_dict['dataset_scores'])
        gr.Markdown(f'### {locale_dict["dataset_scores_table"]}')
        score_table = gr.DataFrame(value=None)

    with gr.Tab(locale_dict['dataset_details']):
        dataset_radio = gr.Radio(label=locale_dict['select_dataset'], choices=[], show_label=True, interactive=True)
        # show dataset details
        with gr.Accordion(locale_dict['report_analysis'], open=True):
            report_analysis = gr.Markdown(value='N/A')
        gr.Markdown(f'### {locale_dict["dataset_scores"]}')
        dataset_plot = gr.Plot(value=None, scale=1, label=locale_dict['dataset_scores'])
        gr.Markdown(f'### {locale_dict["dataset_scores_table"]}')
        dataset_table = gr.DataFrame(value=None)

        gr.Markdown(f'### {locale_dict["model_prediction"]}')
        subset_select = gr.Dropdown(label=locale_dict['select_subset'], choices=[], show_label=True, interactive=True)

        with gr.Row():
            answer_mode_radio = gr.Radio(
                label=locale_dict['answer_mode'], choices=['All', 'Pass', 'Fail'], value='All', interactive=True
            )
            score_threshold = gr.Number(value=0.99, label=locale_dict['score_threshold'], interactive=True)

        data_review_df = gr.State(None)
        filtered_review_df = gr.State(None)

        # show statistics
        with gr.Row(variant='panel'):
            with gr.Column():
                gr.Markdown('### *Counts*')
                answer_mode_counts = gr.Markdown('')
            with gr.Column():
                page_number = gr.Number(
                    value=1, label=locale_dict['page'], minimum=1, maximum=1, step=1, interactive=True
                )

        # show data review table
        with gr.Row(variant='panel'):
            with gr.Column():
                gr.Markdown('### *Score*')
                score_text = gr.Code('', elem_id='score_text', language='json', wrap_lines=False)
            with gr.Column():
                gr.Markdown('### *Normalized Score*')
                nscore = gr.Markdown('', elem_id='score_text', latex_delimiters=LATEX_DELIMITERS)

        with gr.Row(variant='panel'):
            with gr.Column():
                gr.Markdown('### *Gold*')
                gold_text = gr.Markdown('', elem_id='gold_text', latex_delimiters=LATEX_DELIMITERS)
            with gr.Column():
                gr.Markdown('### *Pred*')
                pred_text = gr.Markdown('', elem_id='pred_text', latex_delimiters=LATEX_DELIMITERS)

        with gr.Row(variant='panel'):
            with gr.Column():
                gr.Markdown('### *Input*')
                input_text = gr.Markdown('', elem_id='input_text', latex_delimiters=LATEX_DELIMITERS)
            with gr.Column():
                gr.Markdown('### *Generated*')
                generated_text = gr.Markdown('', elem_id='generated_text', latex_delimiters=LATEX_DELIMITERS)

    @report_name.change(
        inputs=[sidebar.root_path, report_name],
        outputs=[report_list, task_config, dataset_radio, work_dir, model_name]
    )
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
        outputs=[dataset_plot, dataset_table, subset_select, data_review_df, report_analysis]
    )
    def update_single_report_dataset(dataset_name, report_list):
        logger.debug(f'Updating single report dataset: {dataset_name}')
        report_df = get_data_frame(report_list=report_list, flatten_metrics=True, flatten_categories=True)
        analysis = get_report_analysis(report_list, dataset_name)
        data_score_df, styler = get_single_dataset_df(report_df, dataset_name)
        data_score_plot = plot_single_dataset_scores(data_score_df)
        # Only select the subsets that Cat.0 is not '-'
        df_for_subsets = data_score_df.copy()
        subsets = sorted(
            df_for_subsets.loc[df_for_subsets[f'{ReportKey.category_prefix}0'].ne('-'),
                               ReportKey.subset_name].dropna().unique().tolist()
        )

        logger.debug(f'subsets: {subsets}')
        return data_score_plot, styler, gr.update(choices=subsets, value=None), None, analysis

    @gr.on(
        triggers=[subset_select.change],
        inputs=[work_dir, model_name, dataset_radio, subset_select],
        outputs=[data_review_df, page_number]
    )
    def update_single_report_subset(work_dir, model_name, dataset_name, subset_name):
        if not subset_name:
            return gr.skip()
        data_review_df = get_model_prediction(work_dir, model_name, dataset_name, subset_name)
        return data_review_df, 1

    @gr.on(
        triggers=[data_review_df.change, answer_mode_radio.change, score_threshold.change],
        inputs=[data_review_df, answer_mode_radio, score_threshold],
        outputs=[filtered_review_df, page_number, answer_mode_counts]
    )
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
        outputs=[input_text, generated_text, gold_text, pred_text, score_text, nscore]
    )
    def update_table_components(filtered_df, page_number, score_threshold):
        if filtered_df is None or len(filtered_df) == 0:
            return '', '', '', '', '', ''

        # Get single row data for the current page
        start = (page_number - 1)
        if start >= len(filtered_df):
            return '', '', '', '', '', ''

        row = filtered_df.iloc[start]

        # Process the data for display
        input_md = process_model_prediction(row['Input']) + '\n\n' + process_model_prediction(row['Metadata'])
        generated_md = convert_markdown_image(row['Generated'])
        gold_md = convert_markdown_image(row['Gold'])
        pred_md = process_model_prediction(row['Pred'])
        score_md = process_json_content(row['Score'])
        nscore_val = float(row['NScore']) if not pd.isna(row['NScore']) else 0.0

        if nscore_val >= score_threshold:
            nscore_val = f'<div style="background-color:rgb(45,104, 62); padding:10px;">{nscore_val}</div>'
        else:
            nscore_val = f'<div style="background-color:rgb(151, 31, 44); padding:10px;">{nscore_val}</div>'

        return input_md, generated_md, gold_md, pred_md, score_md, nscore_val

    return SingleModelComponents(report_name=report_name)
