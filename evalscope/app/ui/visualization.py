"""
Visualization components for the Evalscope dashboard.
"""
import gradio as gr
from dataclasses import dataclass
from typing import TYPE_CHECKING

from ..utils.localization import get_visualization_locale
from .multi_model import MultiModelComponents, create_multi_model_tab
from .single_model import SingleModelComponents, create_single_model_tab

if TYPE_CHECKING:
    from .sidebar import SidebarComponents


@dataclass
class VisualizationComponents:
    single_model: SingleModelComponents
    multi_model: MultiModelComponents


def create_visualization(sidebar: 'SidebarComponents', lang: str):
    locale_dict = get_visualization_locale(lang)

    with gr.Column(visible=True):
        gr.Markdown(f'## {locale_dict["visualization"]}')
        with gr.Tabs():
            with gr.Tab(locale_dict['single_model']):
                single = create_single_model_tab(sidebar, lang)

            with gr.Tab(locale_dict['multi_model']):
                multi = create_multi_model_tab(sidebar, lang)
    return VisualizationComponents(
        single_model=single,
        multi_model=multi,
    )
