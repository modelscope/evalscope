"""
Multi model components for the Evalscope dashboard.
"""
import gradio as gr
import os
import pandas as pd
from dataclasses import dataclass
from typing import TYPE_CHECKING

from evalscope.report import ReportKey, get_data_frame
from evalscope.utils.logger import get_logger
from ..constants import LATEX_DELIMITERS, MODEL_TOKEN, REPORT_TOKEN
from ..utils.data_utils import (
    get_acc_report_df,
    get_compare_report_df,
    get_model_prediction,
    get_single_dataset_df,
    load_multi_report,
    load_single_report,
)
from ..utils.localization import get_multi_model_locale
from ..utils.text_utils import convert_markdown_image, process_model_prediction
from ..utils.visualization import plot_multi_report_radar

if TYPE_CHECKING:
    from .sidebar import SidebarComponents

logger = get_logger()


@dataclass
class MultiModelComponents:
    multi_report_name: gr.Dropdown


def create_multi_model_tab(sidebar: 'SidebarComponents', lang: str):
    locale_dict = get_multi_model_locale(lang)

    multi_report_name = gr.Dropdown(label=locale_dict['select_reports'], choices=[], multiselect=True, interactive=True)
    report_list = gr.State([])

    with gr.Tab(locale_dict['models_overview']):
        gr.Markdown(locale_dict['model_radar'])
        radar_plot = gr.Plot(value=None)
        gr.Markdown(locale_dict['model_scores'])
        score_table = gr.DataFrame(value=None)

    with gr.Tab(locale_dict['model_comparison_details']):
        with gr.Row():
            model_a_select = gr.Dropdown(label=locale_dict['select_model_a'], choices=[], interactive=True)
            model_b_select = gr.Dropdown(label=locale_dict['select_model_b'], choices=[], interactive=True)

        # States to store selected models' information
        model_a_report = gr.State(None)
        model_b_report = gr.State(None)
        model_a_dir = gr.State(None)
        model_b_dir = gr.State(None)
        model_a_name = gr.State(None)
        model_b_name = gr.State(None)

        dataset_radio = gr.Radio(label=locale_dict['select_dataset'], choices=[], show_label=True, interactive=True)

        gr.Markdown(f"### {locale_dict['model_predictions']}")
        subset_select = gr.Dropdown(label=locale_dict['select_subset'], choices=[], show_label=True, interactive=True)

        with gr.Row():
            answer_mode_radio = gr.Radio(
                label=locale_dict.get('answer_mode'),
                choices=['All', 'Pass A & B', 'Fail A & B', 'Pass A, Fail B', 'Fail A, Pass B'],
                value='All',
                interactive=True
            )
            score_threshold = gr.Number(value=0.99, label=locale_dict['score_threshold'], interactive=True)

        data_comparison_df = gr.State(None)
        filtered_comparison_df = gr.State(None)

        # Statistics row
        with gr.Row(variant='panel'):
            with gr.Column():
                gr.Markdown('### *Counts*')
                comparison_counts = gr.Markdown('')
            with gr.Column():
                page_number = gr.Number(
                    value=1, label=locale_dict['page'], minimum=1, maximum=1, step=1, interactive=True
                )

        # Input and Gold answer sections remain at the top
        with gr.Row(variant='panel'):
            with gr.Column():
                gr.Markdown('### *Input*')
                input_text = gr.Markdown('', elem_id='input_text', latex_delimiters=LATEX_DELIMITERS)

            with gr.Column():
                gr.Markdown('### *Gold Answer*')
                gold_text = gr.Markdown('', elem_id='gold_text', latex_delimiters=LATEX_DELIMITERS)

        # Table-like layout for direct comparison
        with gr.Row():
            # Headers for the two models
            with gr.Column(scale=1):
                gr.Markdown('### *Model A*')
            with gr.Column(scale=1):
                gr.Markdown('### *Model B*')

        # Score comparison row
        with gr.Row():
            with gr.Column(scale=1, variant='panel'):
                gr.Markdown('### *Score*')
                model_a_score = gr.Markdown('', latex_delimiters=LATEX_DELIMITERS)
            with gr.Column(scale=1, variant='panel'):
                gr.Markdown('### *Score*')
                model_b_score = gr.Markdown('', latex_delimiters=LATEX_DELIMITERS)

        # Normalized score comparison row
        with gr.Row():
            with gr.Column(scale=1, variant='panel'):
                gr.Markdown('### *Normalized Score*')
                model_a_nscore = gr.Markdown('', latex_delimiters=LATEX_DELIMITERS)
            with gr.Column(scale=1, variant='panel'):
                gr.Markdown('### *Normalized Score*')
                model_b_nscore = gr.Markdown('', latex_delimiters=LATEX_DELIMITERS)

        # Prediction comparison row
        with gr.Row():
            with gr.Column(scale=1, variant='panel'):
                gr.Markdown('### *Prediction*')
                model_a_pred = gr.Markdown('', latex_delimiters=LATEX_DELIMITERS)
            with gr.Column(scale=1, variant='panel'):
                gr.Markdown('### *Prediction*')
                model_b_pred = gr.Markdown('', latex_delimiters=LATEX_DELIMITERS)

        # Generated output comparison row
        with gr.Row():
            with gr.Column(scale=1, variant='panel'):
                gr.Markdown('### *Generated*')
                model_a_generated = gr.Markdown('', latex_delimiters=LATEX_DELIMITERS)
            with gr.Column(scale=1, variant='panel'):
                gr.Markdown('### *Generated*')
                model_b_generated = gr.Markdown('', latex_delimiters=LATEX_DELIMITERS)

    @multi_report_name.change(
        inputs=[sidebar.root_path, multi_report_name],
        outputs=[report_list, radar_plot, score_table, model_a_select, model_b_select]
    )
    def update_multi_report_data(root_path, multi_report_names):
        if not multi_report_names:
            return gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip()

        report_list = load_multi_report(root_path, multi_report_names)
        report_df, _ = get_acc_report_df(report_list)
        report_radar_plot = plot_multi_report_radar(report_df)
        _, styler = get_compare_report_df(report_df)

        # Extract model names for dropdowns
        model_choices = multi_report_names

        return report_list, report_radar_plot, styler, gr.update(
            choices=model_choices, value=model_choices[0]
        ), gr.update(choices=model_choices, value=model_choices[1] if len(model_choices) > 1 else None)

    @gr.on(
        triggers=[model_a_select.change, model_b_select.change],
        inputs=[sidebar.root_path, model_a_select, model_b_select],
        outputs=[model_a_report, model_b_report, model_a_dir, model_b_dir, model_a_name, model_b_name, dataset_radio]
    )
    def update_selected_models(root_path, model_a, model_b):
        if not model_a or not model_b:
            return gr.skip()

        # Load individual reports for both models
        model_a_reports, datasets_a, _ = load_single_report(root_path, model_a)
        model_b_reports, datasets_b, _ = load_single_report(root_path, model_b)

        # Get common datasets
        common_datasets = list(set(datasets_a).intersection(set(datasets_b)))

        # Extract work directories and model names
        model_a_dir = os.path.join(root_path, model_a.split(REPORT_TOKEN)[0])
        model_b_dir = os.path.join(root_path, model_b.split(REPORT_TOKEN)[0])

        model_a_name = model_a.split(REPORT_TOKEN)[1].split(MODEL_TOKEN)[0]
        model_b_name = model_b.split(REPORT_TOKEN)[1].split(MODEL_TOKEN)[0]

        return (
            model_a_reports, model_b_reports, model_a_dir, model_b_dir, model_a_name, model_b_name,
            gr.update(choices=common_datasets, value=common_datasets[0] if common_datasets else None)
        )

    @gr.on(
        triggers=[dataset_radio.change],
        inputs=[dataset_radio, model_a_report, model_b_report],
        outputs=[subset_select, data_comparison_df]
    )
    def update_dataset_comparison(dataset_name, model_a_report, model_b_report):
        if not dataset_name or model_a_report is None or model_b_report is None:
            return gr.skip()

        # Get dataframes for both models
        report_df_a = get_data_frame(report_list=model_a_report)
        data_score_df_a, _ = get_single_dataset_df(report_df_a, dataset_name)

        report_df_b = get_data_frame(report_list=model_b_report)
        data_score_df_b, _ = get_single_dataset_df(report_df_b, dataset_name)

        # Get subset choices - should be same for both models
        # Only select the subsets that Cat.0 is not '-'
        df_for_subsets = data_score_df_a.copy()
        subsets = sorted(
            df_for_subsets.loc[df_for_subsets[f'{ReportKey.category_prefix}0'].ne('-'),
                               ReportKey.subset_name].dropna().unique().tolist()
        )

        return gr.update(choices=subsets, value=None), None

    @gr.on(
        triggers=[subset_select.change],
        inputs=[model_a_dir, model_b_dir, model_a_name, model_b_name, dataset_radio, subset_select],
        outputs=[data_comparison_df, page_number]
    )
    def update_comparison_data(model_a_dir, model_b_dir, model_a_name, model_b_name, dataset_name, subset_name):
        if not subset_name or not dataset_name:
            return gr.skip()

        # Get predictions for both models
        df_a = get_model_prediction(model_a_dir, model_a_name, dataset_name, subset_name)
        df_b = get_model_prediction(model_b_dir, model_b_name, dataset_name, subset_name)

        # Merge dataframes on Input and Gold columns for comparison
        if df_a is not None and df_b is not None:
            # Save the Index column if it exists
            index_a = df_a['Index'].copy()
            index_b = df_b['Index'].copy()

            df_a = df_a.add_prefix('A_')
            df_b = df_b.add_prefix('B_')

            # Restore the Index column
            df_a['Index'] = index_a
            df_b['Index'] = index_b

            # Merge on Index
            comparison_df = pd.merge(df_a, df_b, on='Index')

            return comparison_df, 1

        return None, 1

    @gr.on(
        triggers=[data_comparison_df.change, answer_mode_radio.change, score_threshold.change],
        inputs=[data_comparison_df, answer_mode_radio, score_threshold],
        outputs=[filtered_comparison_df, page_number, comparison_counts]
    )
    def filter_comparison_data(comparison_df, answer_mode, score_threshold):
        if comparison_df is None:
            return None, gr.update(value=1, maximum=1), ''

        all_count = len(comparison_df)

        # Apply filtering based on the selected mode and threshold
        if answer_mode == 'Pass A & B':
            filtered_df = comparison_df[(comparison_df['A_NScore'] >= score_threshold)
                                        & (comparison_df['B_NScore'] >= score_threshold)]
        elif answer_mode == 'Fail A & B':
            filtered_df = comparison_df[(comparison_df['A_NScore'] < score_threshold)
                                        & (comparison_df['B_NScore'] < score_threshold)]
        elif answer_mode == 'Pass A, Fail B':
            filtered_df = comparison_df[(comparison_df['A_NScore'] >= score_threshold)
                                        & (comparison_df['B_NScore'] < score_threshold)]
        elif answer_mode == 'Fail A, Pass B':
            filtered_df = comparison_df[(comparison_df['A_NScore'] < score_threshold)
                                        & (comparison_df['B_NScore'] >= score_threshold)]
        else:  # All
            filtered_df = comparison_df

        # Count statistics
        pass_a_count = len(comparison_df[comparison_df['A_NScore'] >= score_threshold])
        pass_b_count = len(comparison_df[comparison_df['B_NScore'] >= score_threshold])
        pass_both_count = len(
            comparison_df[(comparison_df['A_NScore'] >= score_threshold)
                          & (comparison_df['B_NScore'] >= score_threshold)]
        )
        fail_both_count = len(
            comparison_df[(comparison_df['A_NScore'] < score_threshold)
                          & (comparison_df['B_NScore'] < score_threshold)]
        )

        counts_text = (
            f'### All: {all_count} | Pass A: {pass_a_count} | Pass B: {pass_b_count} | '
            f'Pass Both: {pass_both_count} | Fail Both: {fail_both_count}'
        )

        max_page = max(1, len(filtered_df))

        return filtered_df, gr.update(value=1, maximum=max_page), counts_text

    @gr.on(
        triggers=[filtered_comparison_df.change, page_number.change, model_a_select.change, model_b_select.change],
        inputs=[
            filtered_comparison_df, page_number, score_threshold, model_a_select, model_b_select, model_a_name,
            model_b_name
        ],
        outputs=[
            input_text, gold_text, model_a_generated, model_a_pred, model_a_score, model_a_nscore, model_b_generated,
            model_b_pred, model_b_score, model_b_nscore
        ]
    )
    def update_comparison_display(
        filtered_df, page_number, score_threshold, model_a_select, model_b_select, model_a_name_val, model_b_name_val
    ):
        if filtered_df is None or len(filtered_df) == 0:
            return '', '', '', '', '', '', '', '', '', ''

        # Get the row for the current page
        start = (page_number - 1)
        if start >= len(filtered_df):
            return '', '', '', '', '', '', '', '', '', ''

        row = filtered_df.iloc[start]

        # Process common data
        input_md = process_model_prediction(row['A_Input'])  # Use A's input (same as B's)
        gold_md = process_model_prediction(row['A_Gold'])  # Use A's gold (same as B's)

        # Process Model A data
        a_generated_md = process_model_prediction(row['A_Generated'])
        a_pred_md = convert_markdown_image(process_model_prediction(row['A_Pred']))
        a_score_md = process_model_prediction(row['A_Score'])
        a_nscore_val = float(row['A_NScore']) if not pd.isna(row['A_NScore']) else 0.0

        # Process Model B data
        b_generated_md = process_model_prediction(row['B_Generated'])
        b_pred_md = convert_markdown_image(process_model_prediction(row['B_Pred']))
        b_score_md = process_model_prediction(row['B_Score'])
        b_nscore_val = float(row['B_NScore']) if not pd.isna(row['B_NScore']) else 0.0

        # Apply visual indicators with backgrounds that make differences more obvious
        if a_nscore_val >= score_threshold:
            a_nscore_html = f"<div style='background-color:rgb(45,104, 62); padding:10px;'>{a_nscore_val}</div>"
        else:
            a_nscore_html = f"<div style='background-color:rgb(151, 31, 44); padding:10px;'>{a_nscore_val}</div>"

        if b_nscore_val >= score_threshold:
            b_nscore_html = f"<div style='background-color:rgb(45,104, 62); padding:10px;'>{b_nscore_val}</div>"
        else:
            b_nscore_html = f"<div style='background-color:rgb(151, 31, 44); padding:10px;'>{b_nscore_val}</div>"

        return (
            input_md, gold_md, a_generated_md, a_pred_md, a_score_md, a_nscore_html, b_generated_md, b_pred_md,
            b_score_md, b_nscore_html
        )

    return MultiModelComponents(multi_report_name=multi_report_name)
