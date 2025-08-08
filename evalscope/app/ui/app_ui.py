"""
Main UI application for the Evalscope dashboard.
"""
import argparse
import gradio as gr

from evalscope.version import __version__
from ..utils.localization import get_app_locale
from .sidebar import create_sidebar
from .visualization import create_visualization


def create_app_ui(args: argparse.Namespace):
    lang = args.lang
    locale_dict = get_app_locale(lang)

    with gr.Blocks(title='Evalscope Dashboard') as demo:
        gr.HTML(f'<h1 style="text-align: left;">{locale_dict["title"]} (v{__version__})</h1>')
        with gr.Row():
            with gr.Column(scale=0, min_width=35):
                toggle_btn = gr.Button('<')
            with gr.Column(scale=1):
                gr.HTML(f'<h3 style="text-align: left;">{locale_dict["star_beggar"]}</h3>')

        with gr.Row():
            with gr.Column(scale=1) as sidebar_column:
                sidebar_visible = gr.State(True)
                sidebar = create_sidebar(args.outputs, lang)

            with gr.Column(scale=5):
                visualization = create_visualization(sidebar, lang)

        @sidebar.load_btn.click(
            inputs=[sidebar.reports_dropdown],
            outputs=[visualization.single_model.report_name, visualization.multi_model.multi_report_name]
        )
        def update_displays(reports_dropdown):
            if not reports_dropdown:
                gr.Warning(locale_dict['note'], duration=3)
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

    return demo
