"""
Main UI application for the Evalscope dashboard.
"""
import argparse
import gradio as gr

from evalscope.utils.logger import get_logger
from evalscope.version import __version__
from ..utils.data_utils import scan_for_report_folders
from ..utils.localization import get_app_locale
from .sidebar import create_sidebar
from .visualization import create_visualization

logger = get_logger()


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

        # Add URL parameter support for reports selection
        @demo.load(
            inputs=[],
            outputs=[
                sidebar.reports_dropdown, visualization.single_model.report_name,
                visualization.multi_model.multi_report_name
            ]
        )
        def load_from_url_params(request: gr.Request):
            """
            Load reports from URL parameters on page load.
            URL format: ?reports=report1;report2;report3
            """
            if request:
                query_params = getattr(request, 'query_params', {})
                reports_param = query_params.get('reports', '')
                logger.debug(f'reports_param from url: {reports_param}')

                if reports_param:
                    # Parse comma-separated report names
                    selected_reports = [r.strip() for r in reports_param.split(';') if r.strip()]
                    logger.debug(f'selected_reports: {selected_reports}')

                    # Get available reports from the outputs directory
                    available_folders = scan_for_report_folders(args.outputs)
                    logger.debug(f'available_folders: {available_folders}')

                    # Filter to only include valid reports by pre-building a prefix map for efficiency.
                    available_folders_set = set(available_folders)
                    prefix_map = {}
                    for folder in available_folders:
                        prefix = folder.split('::')[0]
                        prefix_map.setdefault(prefix, []).append(folder)

                    valid_reports = []
                    for selected in selected_reports:
                        if '::' in selected:
                            # Exact match
                            if selected in available_folders_set:
                                valid_reports.append(selected)
                        else:
                            # Prefix match
                            valid_reports.extend(prefix_map.get(selected, []))
                    # Remove duplicates and maintain the original order
                    valid_reports = list(dict.fromkeys(valid_reports))
                    logger.debug(f'valid_reports: {valid_reports}')

                    if valid_reports:
                        return (
                            gr.update(choices=available_folders,
                                      value=valid_reports), gr.update(choices=valid_reports, value=valid_reports[0]),
                            gr.update(choices=valid_reports, value=valid_reports)
                        )

            # Return default empty state if no valid URL params
            return gr.update(), gr.update(), gr.update()

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
