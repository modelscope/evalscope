"""
Multi model components for the Evalscope dashboard.
"""
import gradio as gr
from dataclasses import dataclass
from typing import TYPE_CHECKING

from evalscope.utils.logger import get_logger
from ..utils.data_utils import get_acc_report_df, get_compare_report_df, load_multi_report
from ..utils.localization import get_multi_model_locale
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
    gr.Markdown(locale_dict['model_radar'])
    radar_plot = gr.Plot(value=None)
    gr.Markdown(locale_dict['model_scores'])
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
