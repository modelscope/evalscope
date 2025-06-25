"""
Sidebar components for the Evalscope dashboard.
"""
import gradio as gr
import os
from dataclasses import dataclass

from evalscope.utils.logger import get_logger
from ..utils.data_utils import scan_for_report_folders
from ..utils.localization import get_sidebar_locale

logger = get_logger()


@dataclass
class SidebarComponents:
    root_path: gr.Textbox
    reports_dropdown: gr.Dropdown
    load_btn: gr.Button


def create_sidebar(outputs_dir: str, lang: str):
    locale_dict = get_sidebar_locale(lang)

    gr.Markdown(f'## {locale_dict["settings"]}')
    root_path = gr.Textbox(label=locale_dict['report_root_path'], value=outputs_dir, placeholder=outputs_dir, lines=1)
    reports_dropdown = gr.Dropdown(label=locale_dict['select_reports'], choices=[], multiselect=True, interactive=True)
    load_btn = gr.Button(locale_dict['load_btn'])
    gr.Markdown(f'### {locale_dict["note"]}')

    @reports_dropdown.focus(inputs=[root_path], outputs=[reports_dropdown])
    def update_dropdown_choices(root_path):
        folders = scan_for_report_folders(root_path)
        if len(folders) == 0:
            gr.Warning(locale_dict['warning'], duration=3)
        return gr.update(choices=folders)

    return SidebarComponents(
        root_path=root_path,
        reports_dropdown=reports_dropdown,
        load_btn=load_btn,
    )
