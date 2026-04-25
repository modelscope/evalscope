"""Blueprint for report browsing and data access.

Exposes the file-system report data through a REST API so that the
React SPA frontend can load reports, predictions and analyses without
direct filesystem access.
"""
import json
import os
from flask import Blueprint, jsonify, request, send_file
from typing import List

from evalscope.app.utils.data_utils import (
    get_acc_report_df,
    get_compare_report_df,
    get_model_prediction,
    get_report_analysis,
    load_multi_report,
    load_single_report,
    normalize_score,
    process_report_name,
    scan_for_report_folders,
)
from evalscope.app.utils.visualization import (
    plot_multi_report_radar,
    plot_single_dataset_scores,
    plot_single_report_scores,
    plot_single_report_sunburst,
)
from evalscope.report import ReportKey, get_data_frame
from evalscope.report.report import Report
from evalscope.utils.io_utils import OutputsStructure
from evalscope.utils.logger import get_logger
from ..utils import OUTPUT_DIR

logger = get_logger()

bp_reports = Blueprint('reports', __name__, url_prefix='/api/v1/reports')

_DEFAULT_ROOT = OUTPUT_DIR


def _root_path() -> str:
    return request.args.get('root_path', _DEFAULT_ROOT)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _report_to_dict(report: Report) -> dict:
    """Serialise a Report to a JSON-friendly dict."""
    return {
        'name': report.name,
        'dataset_name': report.dataset_name,
        'model_name': report.model_name,
        'score': report.score,
        'analysis': report.analysis,
        'metrics': [{
            'name': m.name,
            'num': m.num,
            'score': m.score,
            'categories': [{
                'name': list(c.name) if c.name else [],
                'num': c.num,
                'score': c.score,
                'subsets': [{
                    'name': s.name,
                    'score': s.score,
                    'num': s.num
                } for s in c.subsets],
            } for c in m.categories],
        } for m in report.metrics],
    }


def _df_to_records(df) -> list:
    """Convert a pandas DataFrame to a list of dicts, handling NaN."""
    if df is None or df.empty:
        return []
    return json.loads(df.to_json(orient='records', force_ascii=False))


# ------------------------------------------------------------------
# Endpoints
# ------------------------------------------------------------------


@bp_reports.route('/scan', methods=['GET'])
def scan_reports():
    """Scan the output directory for available report folders.

    Query params:
        root_path (str): directory to scan (default: OUTPUT_DIR)
    """
    try:
        root = _root_path()
        reports = scan_for_report_folders(root)
        return jsonify({'reports': reports}), 200
    except Exception as e:
        logger.error(f'Failed to scan reports: {e}')
        return jsonify({'error': str(e)}), 500


@bp_reports.route('/load', methods=['GET'])
def load_report():
    """Load a single report by name.

    Query params:
        root_path   (str): output root directory
        report_name (str): report identifier
    """
    report_name = request.args.get('report_name')
    if not report_name:
        return jsonify({'error': 'report_name is required'}), 400

    try:
        root = _root_path()
        report_list, datasets, task_cfg = load_single_report(root, report_name)
        return jsonify({
            'report_list': [_report_to_dict(r) for r in report_list],
            'datasets': datasets,
            'task_config': task_cfg,
        }), 200
    except Exception as e:
        logger.error(f'Failed to load report {report_name}: {e}')
        return jsonify({'error': str(e)}), 500


@bp_reports.route('/load_multi', methods=['GET'])
def load_multi():
    """Load multiple reports at once.

    Query params:
        root_path    (str): output root directory
        report_names (str): semicolon-separated report identifiers
    """
    names_raw = request.args.get('report_names', '')
    if not names_raw:
        return jsonify({'error': 'report_names is required'}), 400

    names = [n.strip() for n in names_raw.split(';') if n.strip()]
    try:
        root = _root_path()
        report_list = load_multi_report(root, names)
        return jsonify({
            'report_list': [_report_to_dict(r) for r in report_list],
        }), 200
    except Exception as e:
        logger.error(f'Failed to load multi reports: {e}')
        return jsonify({'error': str(e)}), 500


@bp_reports.route('/dataframe', methods=['GET'])
def get_dataframe():
    """Get report data as a flat JSON table.

    Query params:
        root_path        (str): output root directory
        report_name      (str): report identifier
        type             (str): 'acc' (accuracy overview) | 'compare' (pivot) | 'dataset' (single dataset)
        dataset_name     (str): required when type=dataset
    """
    report_name = request.args.get('report_name')
    if not report_name:
        return jsonify({'error': 'report_name is required'}), 400

    df_type = request.args.get('type', 'acc')
    dataset_name = request.args.get('dataset_name', '')

    try:
        root = _root_path()
        report_list, datasets, _ = load_single_report(root, report_name)
        acc_df, _ = get_acc_report_df(report_list)

        if df_type == 'compare':
            df, _ = get_compare_report_df(acc_df)
        elif df_type == 'dataset':
            if not dataset_name:
                return jsonify({'error': 'dataset_name is required for type=dataset'}), 400
            report_df = get_data_frame(report_list=report_list, flatten_metrics=True, flatten_categories=True)
            from evalscope.app.utils.data_utils import get_single_dataset_df
            df, _ = get_single_dataset_df(report_df, dataset_name)
        else:
            df = acc_df

        return jsonify({
            'columns': list(df.columns),
            'data': _df_to_records(df),
        }), 200
    except Exception as e:
        logger.error(f'Failed to get dataframe: {e}')
        return jsonify({'error': str(e)}), 500


@bp_reports.route('/predictions', methods=['GET'])
def get_predictions():
    """Get model predictions for a given subset.

    Query params:
        root_path    (str): output root directory
        report_name  (str): report identifier
        dataset_name (str): dataset name
        subset_name  (str): subset name
    """
    report_name = request.args.get('report_name')
    dataset_name = request.args.get('dataset_name')
    subset_name = request.args.get('subset_name')

    if not all([report_name, dataset_name, subset_name]):
        return jsonify({'error': 'report_name, dataset_name and subset_name are required'}), 400

    try:
        root = _root_path()
        prefix, model_name, _ = process_report_name(report_name)
        work_dir = os.path.join(root, prefix)
        df = get_model_prediction(work_dir, model_name, dataset_name, subset_name)
        return jsonify({
            'predictions': _df_to_records(df),
        }), 200
    except Exception as e:
        logger.error(f'Failed to get predictions: {e}')
        return jsonify({'error': str(e)}), 500


@bp_reports.route('/analysis', methods=['GET'])
def get_analysis():
    """Get the AI analysis text for a dataset.

    Query params:
        root_path    (str): output root directory
        report_name  (str): report identifier
        dataset_name (str): dataset name
    """
    report_name = request.args.get('report_name')
    dataset_name = request.args.get('dataset_name')

    if not report_name or not dataset_name:
        return jsonify({'error': 'report_name and dataset_name are required'}), 400

    try:
        root = _root_path()
        report_list, _, _ = load_single_report(root, report_name)
        analysis = get_report_analysis(report_list, dataset_name)
        return jsonify({'analysis': analysis}), 200
    except Exception as e:
        logger.error(f'Failed to get analysis: {e}')
        return jsonify({'error': str(e)}), 500


@bp_reports.route('/html', methods=['GET'])
def get_html_report():
    """Serve the HTML report file for a given report.

    Query params:
        root_path   (str): output root directory
        report_name (str): report identifier
    """
    report_name = request.args.get('report_name')
    if not report_name:
        return jsonify({'error': 'report_name is required'}), 400

    try:
        root = _root_path()
        prefix, model_name, _ = process_report_name(report_name)
        report_html = os.path.join(root, prefix, OutputsStructure.REPORTS_DIR, 'report.html')

        if not os.path.exists(report_html):
            return jsonify({'error': f'HTML report not found: {report_html}'}), 404

        return send_file(report_html, mimetype='text/html')
    except Exception as e:
        logger.error(f'Failed to get HTML report: {e}')
        return jsonify({'error': str(e)}), 500


@bp_reports.route('/chart', methods=['GET'])
def get_chart():
    """Generate an interactive Plotly chart as standalone HTML.

    Query params:
        root_path    (str): output root directory
        report_name  (str): report identifier (single report)
        report_names (str): semicolon-separated report identifiers (multi report)
        chart_type   (str): 'scores' | 'sunburst' | 'dataset_scores' | 'radar'
        dataset_name (str): required for chart_type=dataset_scores
    """
    chart_type = request.args.get('chart_type', 'scores')
    root = _root_path()

    try:
        fig = None

        if chart_type == 'radar':
            names_raw = request.args.get('report_names', '')
            names = [n.strip() for n in names_raw.split(';') if n.strip()]
            if not names:
                return jsonify({'error': 'report_names is required for radar'}), 400
            report_list = load_multi_report(root, names)
            acc_df, _ = get_acc_report_df(report_list)
            fig = plot_multi_report_radar(acc_df)
        else:
            report_name = request.args.get('report_name')
            if not report_name:
                return jsonify({'error': 'report_name is required'}), 400
            report_list, datasets, _ = load_single_report(root, report_name)
            acc_df, _ = get_acc_report_df(report_list)

            if chart_type == 'sunburst':
                fig = plot_single_report_sunburst(report_list)
            elif chart_type == 'dataset_scores':
                dataset_name = request.args.get('dataset_name', '')
                if not dataset_name:
                    return jsonify({'error': 'dataset_name is required for dataset_scores'}), 400
                report_df = get_data_frame(report_list=report_list, flatten_metrics=True, flatten_categories=True)
                from evalscope.app.utils.data_utils import get_single_dataset_df
                ds_df, _ = get_single_dataset_df(report_df, dataset_name)
                fig = plot_single_dataset_scores(ds_df)
            else:
                fig = plot_single_report_scores(acc_df)

        if fig is None:
            return '<html><body style="background:#0f172a;color:#94a3b8;display:flex;align-items:center;' \
                   'justify-content:center;height:100vh;font-family:sans-serif;">No data to plot</body></html>', \
                   200, {'Content-Type': 'text/html'}

        html = fig.to_html(full_html=True, include_plotlyjs='cdn', config={'responsive': True})
        return html, 200, {'Content-Type': 'text/html'}

    except Exception as e:
        logger.error(f'Failed to generate chart: {e}')
        return jsonify({'error': str(e)}), 500
