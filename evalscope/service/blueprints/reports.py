"""Blueprint for report browsing and data access.

Exposes the file-system report data through a REST API so that the
React SPA frontend can load reports, predictions and analyses without
direct filesystem access.
"""
import json
import mimetypes
import os
import plotly.express as px
import plotly.graph_objects as go
import shutil
from datetime import datetime
from flask import Blueprint, jsonify, request, send_file
from pydantic import ValidationError
from typing import List, Optional, Tuple

from evalscope.constants import PLOTLY_CDN_URL, PLOTLY_THEME
from evalscope.metrics.semantics import PrimaryMetricRef
from evalscope.metrics.semantics.ranking import bounded_quality_ratio
from evalscope.report import ReportKey, ReportRef, get_data_frame
from evalscope.report.report import Report
from evalscope.report.visualization import (
    plot_multi_report_radar,
    plot_single_dataset_scores,
    plot_single_report_scores,
    plot_single_report_sunburst,
)
from evalscope.utils.data_utils import (
    get_acc_report_df,
    get_compare_report_df,
    get_comparison_quality_report_df,
    get_model_prediction,
    get_quality_metric_df,
    get_quality_report_df,
    get_report_analysis,
    load_multi_report_groups,
    load_report_bundle,
    normalize_score,
    scan_report_refs,
)
from evalscope.utils.io_utils import OutputsStructure
from evalscope.utils.logger import get_logger
from ..utils import OUTPUT_DIR, active_task_ids

logger = get_logger()

bp_reports = Blueprint('reports', __name__, url_prefix='/api/v1/reports')

_DEFAULT_ROOT = OUTPUT_DIR

# Allowed extensions for the media proxy (security: do not serve arbitrary files)
_MEDIA_EXTENSIONS = {
    # image
    '.jpg',
    '.jpeg',
    '.png',
    '.gif',
    '.webp',
    '.bmp',
    '.svg',
    '.ico',
    # video
    '.mp4',
    '.webm',
    '.ogg',
    '.ogv',
    '.mov',
    '.avi',
    '.mkv',
    # audio
    '.mp3',
    '.wav',
    '.flac',
    '.aac',
    '.m4a',
    '.opus',
}


@bp_reports.route('/media/file', methods=['GET'])
def serve_media_file():
    """Serve a local media file (image / audio / video) via HTTP.

    This proxy endpoint allows the browser to load server-side local file
    paths that are stored inside prediction records (e.g. video paths from
    MVBench datasets).

    Query params:
        path (str): Absolute path to the media file on the server.

    Security:
        - Only files with known media extensions are served.
        - The file must exist and be a regular file.
    """
    file_path = request.args.get('path', '').strip()
    if not file_path:
        return jsonify({'error': 'path parameter is required'}), 400

    # Normalise to absolute path and reject directory traversal
    file_path = os.path.realpath(file_path)

    ext = os.path.splitext(file_path)[1].lower()
    if ext not in _MEDIA_EXTENSIONS:
        return jsonify({'error': f'File type {ext!r} is not allowed'}), 403

    if not os.path.isfile(file_path):
        return jsonify({'error': 'File not found'}), 404

    mime_type, _ = mimetypes.guess_type(file_path)
    if mime_type is None:
        mime_type = 'application/octet-stream'

    return send_file(file_path, mimetype=mime_type)


def _root_path() -> str:
    # Priority: URL query param > app config (from --outputs CLI arg) > default
    from flask import current_app
    return request.args.get('root_path', current_app.config.get('OUTPUTS_ROOT') or _DEFAULT_ROOT)


def _apply_chart_theme(fig: go.Figure, theme: str) -> None:
    """Apply the Web console theme to a generated Plotly figure."""
    template = 'plotly_white' if theme == 'light' else PLOTLY_THEME
    fig.update_layout(template=template)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _df_to_records(df) -> list:
    """Convert a pandas DataFrame to a list of dicts, handling NaN."""
    if df is None or df.empty:
        return []
    return json.loads(df.to_json(orient='records', force_ascii=False))


def _parse_path_ref(run_id: str, model_id: str) -> ReportRef:
    """Build a reference from URL path segments.

    Raises:
        ValidationError: when a segment is not a single, non-escaping name.
    """
    return ReportRef(run_id=run_id, model_id=model_id)


def _query_refs() -> List[ReportRef]:
    """Parse the repeated ``report={run_id}/{model_id}`` query parameter.

    Raises:
        ValueError: when a value is not a valid flat reference.
    """
    return [ReportRef.parse(value.strip()) for value in request.args.getlist('report') if value.strip()]


def _extract_timestamp(ref: ReportRef, root: str) -> str:
    """Try to extract a timestamp from the run directory name or fall back to mtime."""
    # Directory names typically look like "20260423_201338"
    for fmt in ('%Y%m%d_%H%M%S', '%Y%m%d'):
        try:
            return datetime.strptime(ref.run_id, fmt).isoformat()
        except ValueError:
            continue
    # Fall back to directory modification time
    run_dir = os.path.join(root, ref.run_id)
    if os.path.isdir(run_dir):
        return datetime.fromtimestamp(os.path.getmtime(run_dir)).isoformat()
    return ''


def _build_report_meta(ref: ReportRef, root: str) -> Optional[dict]:
    """Load a report and return lightweight metadata for the list endpoint."""
    try:
        report_list, _, _ = load_report_bundle(root, ref)
    except Exception:
        return None

    if not report_list:
        return None

    # Aggregate: use the first report's model_name; collect all dataset names
    first = report_list[0]
    total_num = 0
    dataset_names = []
    dataset_pretty_names = []
    primary_metrics = []
    for r in report_list:
        dataset_names.append(r.dataset_name)
        dataset_pretty_names.append(r.dataset_pretty_name or r.dataset_name)
        total_num += r.num or 0
        primary_metric = r.primary_metric
        primary_metrics.append(primary_metric)
    timestamp = _extract_timestamp(ref, root)

    # Every selected primary contributes a reference so it can be shown in its native scale.
    # Multiple datasets are never collapsed into a single cross-benchmark ranking number.
    primary_metric_refs = [
        PrimaryMetricRef(
            dataset_name=r.dataset_name,
            dataset_pretty_name=r.dataset_pretty_name or r.dataset_name,
            identity=metric.identity,
            score=metric.score,
            semantics=metric.semantics,
        ) for r, metric in zip(report_list, primary_metrics) if metric
    ]
    quality_ratio = None
    if len(report_list) == 1 and len(primary_metric_refs) == 1:
        metric = primary_metrics[0]
        quality_ratio = bounded_quality_ratio(metric.score, metric.semantics)

    return {
        'run_id': ref.run_id,
        'model_id': ref.model_id,
        'model_name': first.model_name,
        'dataset_name': ', '.join(dataset_names) if len(dataset_names) > 1 else
        (dataset_names[0] if dataset_names else ''),
        'dataset_pretty_name': ', '.join(dataset_pretty_names) if len(dataset_pretty_names) > 1 else
        (dataset_pretty_names[0] if dataset_pretty_names else ''),
        'num_samples': total_num,
        'timestamp': timestamp,
        'primary_metrics': [ref.model_dump(mode='json') for ref in primary_metric_refs],
        # Ranking and filtering key only, never rendered: a direction-aware 0-1 quality ratio, so
        # a low WER ranks as well as a high accuracy and a 0-100 scale is not treated as 100x
        # better than a ratio. `None` means the run's metrics admit no such scale.
        'quality_ratio': quality_ratio,
        # keep individual scores for per-dataset filtering
        '_datasets': dataset_names,
    }


def _report_to_service_dict(report: Report) -> dict:
    """Serialize the persisted Report v2 contract without boundary-time reinjection."""
    return report.to_dict()


def _refresh_html_report(reports_dir: str) -> None:
    """Refresh a generated report.html, or remove it if regeneration fails."""
    html_report = os.path.join(reports_dir, 'report.html')
    if not os.path.isfile(html_report):
        return

    try:
        from evalscope.report import gen_html_report_file
        gen_html_report_file(reports_dir)
    except Exception as e:
        logger.warning(f'Failed to refresh report HTML after deletion, removing stale file: {e}')
        try:
            os.remove(html_report)
        except OSError as remove_error:
            logger.warning(f'Failed to remove stale report HTML {html_report}: {remove_error}')


# ------------------------------------------------------------------
# Endpoints
# ------------------------------------------------------------------


@bp_reports.route('', methods=['GET'])
def list_reports():
    """Return a filterable, paginated list of reports with metadata.

    Query params:
        root_path  (str):   output root directory (required)
        search     (str):   fuzzy search on model/dataset name
        models     (str):   semicolon-separated model filter
        datasets   (str):   semicolon-separated dataset filter
        score_min  (float): minimum score (0-1)
        score_max  (float): maximum score (0-1)
        sort_by    (str):   score / model / dataset / time (default: time)
        sort_order (str):   asc / desc (default: desc)
        page       (int):   page number (default: 1)
        page_size  (int):   items per page (default: 20)
    """
    try:
        root = _root_path()
        if not root or not os.path.isdir(root):
            return jsonify({'error': 'root_path is required and must be an existing directory'}), 400

        # --- Scan & load metadata ---
        items = []
        for ref in scan_report_refs(root):
            meta = _build_report_meta(ref, root)
            if meta is not None:
                items.append(meta)

        # Collect available filter values before filtering
        available_models = sorted({it['model_name'] for it in items})
        available_datasets = sorted({ds for it in items for ds in it['_datasets']})

        # --- Filters ---
        search = request.args.get('search', '').strip().lower()
        if search:
            items = [
                it for it in items if search in it['model_name'].lower() or search in it['dataset_name'].lower()
                or search in it['dataset_pretty_name'].lower()
            ]

        models_filter = request.args.get('models', '').strip()
        if models_filter:
            model_set = {m.strip().lower() for m in models_filter.split(';') if m.strip()}
            items = [it for it in items if it['model_name'].lower() in model_set]

        datasets_filter = request.args.get('datasets', '').strip()
        if datasets_filter:
            ds_set = {d.strip().lower() for d in datasets_filter.split(';') if d.strip()}
            items = [it for it in items if any(d.lower() in ds_set for d in it['_datasets'])]

        # The range is expressed as a 0-1 quality ratio, not as a raw score. Filtering raw values
        # against a fixed 0-1 window silently dropped every benchmark on an official 0-100 scale:
        # arena_hard reporting 87.25 could never satisfy `score <= 1`. A run whose metrics are not
        # rankable has no ratio and is excluded whenever a quality range is active.
        score_min = request.args.get('score_min', type=float)
        score_max = request.args.get('score_max', type=float)
        if score_min is not None:
            items = [it for it in items if it['quality_ratio'] is not None and it['quality_ratio'] >= score_min]
        if score_max is not None:
            items = [it for it in items if it['quality_ratio'] is not None and it['quality_ratio'] <= score_max]

        # --- Sort ---
        sort_by = request.args.get('sort_by', 'time')
        sort_order = request.args.get('sort_order', 'desc')
        reverse = sort_order == 'desc'

        sort_key_map = {
            'model': lambda x: x['model_name'].lower(),
            'dataset': lambda x: x['dataset_name'].lower(),
            'time': lambda x: x['timestamp'],
        }
        if sort_by == 'score':
            rankable = [item for item in items if item['quality_ratio'] is not None]
            unrankable = [item for item in items if item['quality_ratio'] is None]
            rankable.sort(key=lambda item: item['quality_ratio'], reverse=reverse)
            items = rankable + unrankable
        else:
            key_fn = sort_key_map.get(sort_by, sort_key_map['time'])
            items.sort(key=key_fn, reverse=reverse)

        # --- Paginate ---
        page = max(1, request.args.get('page', 1, type=int))
        page_size = max(1, min(100, request.args.get('page_size', 20, type=int)))
        total = len(items)
        start = (page - 1) * page_size
        page_items = items[start:start + page_size]

        # Strip internal keys before returning
        for it in page_items:
            it.pop('_datasets', None)

        return jsonify({
            'reports': page_items,
            'total': total,
            'page': page,
            'page_size': page_size,
            'filters': {
                'available_models': available_models,
                'available_datasets': available_datasets,
            },
        }), 200

    except Exception as e:
        logger.error(f'Failed to list reports: {e}')
        return jsonify({'error': str(e)}), 500


@bp_reports.route('/runs/<run_id>/models/<model_id>', methods=['DELETE'])
def delete_report(run_id: str, model_id: str):
    """Delete one evaluation report (its per-model artefacts) from disk.

    Removes ``<run_id>/{reports,predictions,reviews}/<model_id>``. When no other
    model report remains in the run directory afterwards, the whole
    ``<run_id>/`` directory (logs/configs included) is removed as well.

    Query params:
        root_path (str): output root directory (optional; falls back to config)

    Returns 409 when the report belongs to a task that is still executing.
    """
    try:
        ref = _parse_path_ref(run_id, model_id)
    except ValidationError as e:
        return jsonify({'error': f'Invalid report reference: {e}'}), 400

    # Running-task protection: in the service layout the run directory is the
    # task_id itself, so refuse deletion while that task is still active.
    if ref.run_id in active_task_ids():
        return jsonify({'error': f'Task is still running: {ref.run_id}'}), 409

    root_real = os.path.realpath(_root_path())
    run_dir = os.path.realpath(os.path.join(root_real, ref.run_id))
    # Reject path traversal and symlinks escaping the outputs root.
    if run_dir == root_real or not run_dir.startswith(root_real + os.sep):
        return jsonify({'error': 'Invalid report path'}), 400

    model_report_dir = os.path.realpath(os.path.join(run_dir, OutputsStructure.REPORTS_DIR, ref.model_id))
    if not model_report_dir.startswith(run_dir + os.sep) or not os.path.isdir(model_report_dir):
        return jsonify({'error': f'Report not found: {ref.key}'}), 404

    try:
        for sub in (OutputsStructure.REPORTS_DIR, OutputsStructure.PREDICTIONS_DIR, OutputsStructure.REVIEWS_DIR):
            target = os.path.join(run_dir, sub, ref.model_id)
            if os.path.isdir(target):
                shutil.rmtree(target)
        # Drop the whole run directory once its last model report is gone;
        # logs/configs are run-level artefacts with no report left to serve.
        reports_dir = os.path.join(run_dir, OutputsStructure.REPORTS_DIR)
        has_reports = os.path.isdir(reports_dir) and any(
            os.path.isdir(os.path.join(reports_dir, name)) for name in os.listdir(reports_dir)
        )
        if not has_reports:
            shutil.rmtree(run_dir)
        else:
            _refresh_html_report(reports_dir)
        logger.info(f'Deleted eval report {ref.key} under {run_dir}')
        return jsonify({'success': True, 'run_id': ref.run_id, 'model_id': ref.model_id}), 200
    except Exception as e:
        logger.error(f'Failed to delete report {ref.key}: {e}')
        return jsonify({'error': str(e)}), 500


@bp_reports.route('/runs/<run_id>/models/<model_id>', methods=['GET'])
def load_report(run_id: str, model_id: str):
    """Load one model report of one run.

    Query params:
        root_path (str): output root directory
    """
    try:
        ref = _parse_path_ref(run_id, model_id)
    except ValidationError as e:
        return jsonify({'error': f'Invalid report reference: {e}'}), 400

    try:
        report_list, datasets, task_cfg = load_report_bundle(_root_path(), ref)
        return jsonify({
            'report_list': [_report_to_service_dict(r) for r in report_list],
            'datasets': datasets,
            'task_config': task_cfg,
        }), 200
    except FileNotFoundError as e:
        logger.warning(f'Report {ref.key} not found: {e}')
        return jsonify({'error': f'Report not found: {ref.key}'}), 404
    except Exception as e:
        logger.error(f'Failed to load report {ref.key}: {e}')
        return jsonify({'error': str(e)}), 500


@bp_reports.route('/runs/<run_id>/models/<model_id>/table', methods=['GET'])
def get_dataframe(run_id: str, model_id: str):
    """Get report data as a flat JSON table.

    Query params:
        root_path    (str): output root directory
        view         (str): 'acc' (accuracy overview) | 'compare' (pivot) | 'dataset' (single dataset)
        dataset_name (str): required when view=dataset
    """
    try:
        ref = _parse_path_ref(run_id, model_id)
    except ValidationError as e:
        return jsonify({'error': f'Invalid report reference: {e}'}), 400

    view = request.args.get('view', 'acc')
    dataset_name = request.args.get('dataset_name', '')

    try:
        report_list, _, _ = load_report_bundle(_root_path(), ref)
        acc_df = get_acc_report_df(report_list)

        if view == 'compare':
            df = get_compare_report_df(acc_df)
        elif view == 'dataset':
            if not dataset_name:
                return jsonify({'error': 'dataset_name is required for view=dataset'}), 400
            report_df = get_data_frame(report_list=report_list, flatten_metrics=True, flatten_categories=True)
            from evalscope.utils.data_utils import get_single_dataset_df
            df = get_single_dataset_df(report_df, dataset_name)
        else:
            df = acc_df

        return jsonify({
            'columns': list(df.columns),
            'data': _df_to_records(df),
        }), 200
    except Exception as e:
        logger.error(f'Failed to get dataframe: {e}')
        return jsonify({'error': str(e)}), 500


@bp_reports.route('/runs/<run_id>/models/<model_id>/predictions', methods=['GET'])
def get_predictions(run_id: str, model_id: str):
    """Get model predictions for a given subset.

    Query params:
        root_path    (str): output root directory
        dataset_name (str): dataset name
        subset_name  (str): subset name
    """
    try:
        ref = _parse_path_ref(run_id, model_id)
    except ValidationError as e:
        return jsonify({'error': f'Invalid report reference: {e}'}), 400

    dataset_name = request.args.get('dataset_name')
    subset_name = request.args.get('subset_name')
    if not dataset_name or not subset_name:
        return jsonify({'error': 'dataset_name and subset_name are required'}), 400

    try:
        work_dir = os.path.join(_root_path(), ref.run_id)
        df = get_model_prediction(work_dir, ref.model_id, dataset_name, subset_name)
        return jsonify({
            'predictions': _df_to_records(df),
        }), 200
    except Exception as e:
        logger.error(f'Failed to get predictions: {e}')
        return jsonify({'error': str(e)}), 500


@bp_reports.route('/runs/<run_id>/models/<model_id>/analysis', methods=['GET'])
def get_analysis(run_id: str, model_id: str):
    """Get the AI analysis text for a dataset.

    Query params:
        root_path    (str): output root directory
        dataset_name (str): dataset name
    """
    try:
        ref = _parse_path_ref(run_id, model_id)
    except ValidationError as e:
        return jsonify({'error': f'Invalid report reference: {e}'}), 400

    dataset_name = request.args.get('dataset_name')
    if not dataset_name:
        return jsonify({'error': 'dataset_name is required'}), 400

    try:
        report_list, _, _ = load_report_bundle(_root_path(), ref)
        analysis = get_report_analysis(report_list, dataset_name)
        return jsonify({'analysis': analysis}), 200
    except Exception as e:
        logger.error(f'Failed to get analysis: {e}')
        return jsonify({'error': str(e)}), 500


@bp_reports.route('/runs/<run_id>/html', methods=['GET'])
def get_html_report(run_id: str):
    """Serve the generated HTML report of a run.

    The HTML report covers every model of the run, so it is addressed by run alone.

    Query params:
        root_path (str): output root directory
    """
    try:
        # Reuse the reference validation by pairing the run with a placeholder model segment.
        ReportRef(run_id=run_id, model_id='_')
    except ValidationError as e:
        return jsonify({'error': f'Invalid run id: {e}'}), 400

    try:
        root = os.path.abspath(_root_path())
        report_html = os.path.join(root, run_id, OutputsStructure.REPORTS_DIR, 'report.html')

        if not os.path.exists(report_html):
            return jsonify({
                'error': 'Report not yet generated',
                'message': 'The HTML report has not been generated for this evaluation. It may still be in progress.',
            }), 404

        return send_file(report_html, mimetype='text/html')
    except Exception as e:
        logger.error(f'Failed to get HTML report: {e}')
        return jsonify({'error': str(e)}), 500


def _render_chart_html(fig: Optional[go.Figure]) -> Tuple[str, int, dict]:
    """Turn a figure into standalone Plotly HTML, or an empty-state page when there is nothing to plot."""
    if fig is None:
        return '<html><body style="background:#0f172a;color:#94a3b8;display:flex;align-items:center;' \
               'justify-content:center;height:100vh;font-family:sans-serif;">No data to plot</body></html>', \
               200, {'Content-Type': 'text/html'}

    _apply_chart_theme(fig, request.args.get('theme', 'dark'))
    html = fig.to_html(full_html=True, include_plotlyjs=False, config={'responsive': True})
    plotly_script = f'<script src="{PLOTLY_CDN_URL}" charset="utf-8"></script>'
    html = html.replace('</head>', f'  {plotly_script}\n</head>')
    return html, 200, {'Content-Type': 'text/html'}


@bp_reports.route('/charts/<chart_type>', methods=['GET'])
def get_compare_chart(chart_type: str):
    """Generate a multi-report comparison chart as standalone HTML.

    Query params:
        root_path (str): output root directory
        report    (str): repeated ``{run_id}/{model_id}`` reference (one per compared report)
        theme     (str): 'light' | 'dark'; defaults to the existing dark report theme
    """
    if chart_type not in ('radar', 'grouped_bar'):
        return jsonify({'error': f'Unknown comparison chart type: {chart_type}'}), 400

    try:
        refs = _query_refs()
    except ValueError as e:
        return jsonify({'error': f'Invalid report reference: {e}'}), 400
    if not refs:
        return jsonify({'error': 'at least one report is required'}), 400

    try:
        quality_df = get_comparison_quality_report_df(load_multi_report_groups(_root_path(), refs))
        fig = None
        if chart_type == 'radar':
            fig = plot_multi_report_radar(quality_df)
        elif not quality_df.empty:
            color_seq = ['#816DF8', '#0F9C7E', '#fbbf24', '#a78bfa', '#63b3ed']
            fig = px.bar(
                quality_df,
                x=ReportKey.model_name,
                y=ReportKey.score,
                color=ReportKey.dataset_name,
                barmode='group',
                text=ReportKey.display_score,
                custom_data=[ReportKey.metric_name, ReportKey.raw_score],
                color_discrete_sequence=color_seq,
            )
            fig.update_traces(
                textposition='outside',
                hovertemplate=(
                    '%{x}<br>%{customdata[0]}: %{text}<br>Quality: %{y:.3f}'
                    '<extra>%{fullData.name}</extra>'
                ),
            )
            fig.update_layout(
                template=PLOTLY_THEME,
                uniformtext_minsize=12,
                uniformtext_mode='hide',
                yaxis=dict(range=[0, 1]),
                margin=dict(t=20, l=20, r=20, b=20),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
            )
        return _render_chart_html(fig)
    except Exception as e:
        logger.error(f'Failed to generate comparison chart: {e}')
        return jsonify({'error': str(e)}), 500


@bp_reports.route('/runs/<run_id>/models/<model_id>/charts/<chart_type>', methods=['GET'])
def get_chart(run_id: str, model_id: str, chart_type: str):
    """Generate a single-report chart as standalone HTML.

    Path params:
        chart_type (str): 'scores' | 'sunburst' | 'dataset_scores' | 'histogram'

    Query params:
        root_path    (str): output root directory
        dataset_name (str): required for the 'dataset_scores' and 'histogram' charts
        subset_name  (str): required for the 'histogram' chart
        theme        (str): 'light' | 'dark'; defaults to the existing dark report theme
    """
    try:
        ref = _parse_path_ref(run_id, model_id)
    except ValidationError as e:
        return jsonify({'error': f'Invalid report reference: {e}'}), 400

    root = _root_path()
    try:
        fig = None
        if chart_type == 'histogram':
            # Score distribution histogram from prediction NScore values
            dataset_name = request.args.get('dataset_name', '')
            subset_name = request.args.get('subset_name', '')
            if not dataset_name or not subset_name:
                return jsonify({'error': 'dataset_name and subset_name are required for histogram'}), 400
            work_dir = os.path.join(root, ref.run_id)
            pred_df = get_model_prediction(work_dir, ref.model_id, dataset_name, subset_name)
            if pred_df is not None and not pred_df.empty and 'NScore' in pred_df.columns:
                fig = px.histogram(
                    pred_df,
                    x='NScore',
                    nbins=20,
                    color_discrete_sequence=['#816DF8'],
                )
                fig.update_layout(
                    template=PLOTLY_THEME,
                    xaxis_title='Score',
                    yaxis_title='Count',
                    margin=dict(t=20, l=20, r=20, b=20),
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                )
        else:
            report_list, _, _ = load_report_bundle(root, ref)
            if chart_type == 'sunburst':
                fig = plot_single_report_sunburst(report_list)
            elif chart_type == 'dataset_scores':
                dataset_name = request.args.get('dataset_name', '')
                if not dataset_name:
                    return jsonify({'error': 'dataset_name is required for dataset_scores'}), 400
                report_df = get_data_frame(report_list=report_list, flatten_metrics=True, flatten_categories=True)
                from evalscope.utils.data_utils import get_single_dataset_df
                ds_df = get_single_dataset_df(report_df, dataset_name)
                fig = plot_single_dataset_scores(get_quality_metric_df(report_list, ds_df))
            elif chart_type == 'scores':
                fig = plot_single_report_scores(get_quality_report_df(report_list))
            else:
                return jsonify({'error': f'Unknown chart type: {chart_type}'}), 400

        return _render_chart_html(fig)
    except Exception as e:
        logger.error(f'Failed to generate chart: {e}')
        return jsonify({'error': str(e)}), 500
