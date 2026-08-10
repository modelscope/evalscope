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
from typing import List

from evalscope.constants import PLOTLY_CDN_URL, PLOTLY_THEME
from evalscope.metrics.semantics import PrimaryMetricRef
from evalscope.metrics.semantics.perf import resolve_perf_semantics
from evalscope.metrics.semantics.ranking import bounded_quality_ratio, mean_quality_ratio
from evalscope.report import ReportKey, get_data_frame
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
    get_model_prediction,
    get_quality_metric_df,
    get_quality_report_df,
    get_report_analysis,
    load_multi_report,
    load_single_report,
    normalize_score,
    process_report_name,
    scan_for_report_folders,
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


def _extract_timestamp(report_name: str, root: str) -> str:
    """Try to extract a timestamp from the report directory name or fall back to mtime."""
    try:
        prefix, _, _ = process_report_name(report_name)
        # Directory names typically look like "20260423_201338"
        for fmt in ('%Y%m%d_%H%M%S', '%Y%m%d'):
            try:
                dt = datetime.strptime(prefix, fmt)
                return dt.isoformat()
            except ValueError:
                continue
        # Fall back to directory modification time
        dir_path = os.path.join(root, prefix)
        if os.path.isdir(dir_path):
            mtime = os.path.getmtime(dir_path)
            return datetime.fromtimestamp(mtime).isoformat()
    except Exception:
        pass
    return ''


def _build_report_meta(report_name: str, root: str) -> dict:
    """Load a report and return lightweight metadata for the list endpoint."""
    try:
        report_list, datasets, _ = load_single_report(root, report_name)
    except Exception:
        return None

    if not report_list:
        return None

    # Aggregate: use the first report's model_name; collect all dataset names
    first = report_list[0]
    total_num = 0
    dataset_names = []
    score_sum = 0.0
    primary_metrics = []
    for r in report_list:
        dataset_names.append(r.dataset_name)
        total_num += r.num or 0
        primary_metric = r.primary_metric
        primary_metrics.append(primary_metric)
        score_sum += primary_metric.score if primary_metric else r.score

    avg_score = round(score_sum / len(report_list), 4) if report_list else 0.0
    timestamp = _extract_timestamp(report_name, root)
    metric_names = [metric.name for metric in primary_metrics if metric]
    metric_name = metric_names[0] if len(metric_names) == len(report_list) and all(
        name == metric_names[0] for name in metric_names
    ) else ''

    # Preserve each metric's native scale; consumers use metric_name to format it.
    dataset_scores = {}
    for r, primary_metric in zip(report_list, primary_metrics):
        score = primary_metric.score if primary_metric else r.score
        dataset_scores[r.dataset_name] = round(score, 4) if score is not None else None

    # Every dataset contributes a reference so each one can be shown as
    # `dataset -> metric -> score`, each in its own native scale. They are never collapsed into a
    # single cross-benchmark number. A report always resolves a primary metric when it has any
    # metric at all, inferring one if the benchmark declared none.
    primary_metric_refs = [
        PrimaryMetricRef(
            dataset_name=r.dataset_name,
            metric_name=metric.name,
            score=metric.score,
            semantics=metric.semantics,
            inferred=r.primary_metric_is_inferred(),
        ) for r, metric in zip(report_list, primary_metrics) if metric
    ]

    return {
        'name': report_name,
        'model_name': first.model_name,
        'dataset_name': ', '.join(dataset_names) if len(dataset_names) > 1 else
        (dataset_names[0] if dataset_names else ''),
        # Deprecated: an average over possibly heterogeneous metrics. Kept for old clients;
        # semantic consumers read `primary_metrics` and `summary_*` instead.
        'score': avg_score,
        'metric_name': metric_name,
        'dataset_scores': dataset_scores,
        'num_samples': total_num,
        'timestamp': timestamp,
        'primary_metrics': [ref.model_dump(mode='json') for ref in primary_metric_refs],
        # Ranking and filtering key only, never rendered: a direction-aware 0-1 quality ratio, so
        # a low WER ranks as well as a high accuracy and a 0-100 scale is not treated as 100x
        # better than a ratio. `None` means the run's metrics admit no such scale.
        'quality_ratio': mean_quality_ratio(
            bounded_quality_ratio(metric.score, metric.semantics) for metric in primary_metrics if metric
        ),
        # keep individual scores for per-dataset filtering
        '_datasets': dataset_names,
        '_scores': [metric.score if metric else report.score for report, metric in zip(report_list, primary_metrics)],
    }


#: Stable API paths of the perf data embedded in a report, used to key its semantics map.
_IN_REPORT_PERF_FIELDS = (
    'n_samples',
    'latency',
    'ttft',
    'tpot',
    'throughput.avg_output_tps',
    'throughput.avg_req_ps',
    'usage.input_tokens',
    'usage.output_tokens',
    'usage.total_tokens',
)


def _report_to_service_dict(report: Report) -> dict:
    """Serialize a report, adding the resolved semantics next to the legacy fields.

    ``score`` and ``metric_name`` keep their existing behaviour for old clients. New clients
    select the primary metric through ``primary_metric_name`` or the per-metric ``role``, never
    through ``score`` or ``metrics[0]``.
    """
    data = report.to_dict()
    if report.primary_metric:
        data['score'] = report.primary_metric.score

    # `Metric.semantics` is excluded from serialization by design (only the anchor is stored),
    # so attach the hydrated contract explicitly for the wire format.
    semantics_by_name = {metric.name: metric.semantics for metric in report.metrics}
    for metric_data in data.get('metrics', []):
        semantics = semantics_by_name.get(metric_data.get('name'))
        metric_data['semantics'] = semantics.model_dump(mode='json') if semantics else None

    # In-report perf data is keyed by stable API paths, so its semantics map uses those keys.
    perf_metrics = data.get('perf_metrics')
    if isinstance(perf_metrics, dict):
        perf_metrics['metric_semantics'] = resolve_perf_semantics(_IN_REPORT_PERF_FIELDS)
    return data


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


@bp_reports.route('/list', methods=['GET'])
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
        raw_reports = scan_for_report_folders(root)
        items = []
        for rn in raw_reports:
            meta = _build_report_meta(rn, root)
            if meta is not None:
                items.append(meta)

        # Collect available filter values before filtering
        available_models = sorted({it['model_name'] for it in items})
        available_datasets = sorted({ds for it in items for ds in it['_datasets']})

        # --- Filters ---
        search = request.args.get('search', '').strip().lower()
        if search:
            items = [it for it in items if search in it['model_name'].lower() or search in it['dataset_name'].lower()]

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
        # rankable has no ratio and is never excluded, since the filter cannot judge it.
        score_min = request.args.get('score_min', type=float)
        score_max = request.args.get('score_max', type=float)
        if score_min is not None:
            items = [it for it in items if it['quality_ratio'] is None or it['quality_ratio'] >= score_min]
        if score_max is not None:
            items = [it for it in items if it['quality_ratio'] is None or it['quality_ratio'] <= score_max]

        # --- Sort ---
        sort_by = request.args.get('sort_by', 'time')
        sort_order = request.args.get('sort_order', 'desc')
        reverse = sort_order == 'desc'

        sort_key_map = {
            # Rank by quality, so a lower-is-better metric is not sorted backwards and metrics on
            # different scales are comparable. Unrankable runs have no ratio and sort at the
            # bottom of whichever direction was asked for.
            'score': lambda x: (x['quality_ratio'] is not None, x['quality_ratio'] or 0.0)
            if reverse else (x['quality_ratio'] is None, x['quality_ratio'] or 0.0),
            'model': lambda x: x['model_name'].lower(),
            'dataset': lambda x: x['dataset_name'].lower(),
            'time': lambda x: x['timestamp'],
        }
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
            it.pop('_scores', None)

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


@bp_reports.route('/report', methods=['DELETE'])
def delete_report():
    """Delete one evaluation report (its per-model artefacts) from disk.

    Removes ``<prefix>/{reports,predictions,reviews}/<model>``. When no other
    model report remains in the run directory afterwards, the whole
    ``<prefix>/`` directory (logs/configs included) is removed as well.

    Query params:
        root_path   (str): output root directory (optional; falls back to config)
        report_name (str): report identifier ``{prefix}@@{model}::{datasets}``

    Returns 409 when the report belongs to a task that is still executing.
    """
    report_name = request.args.get('report_name')
    if not report_name:
        return jsonify({'error': 'report_name is required'}), 400

    try:
        prefix, model_name, _ = process_report_name(report_name)
    except ValueError:
        return jsonify({'error': f'Invalid report_name: {report_name}'}), 400

    # Running-task protection: in the service layout the run directory is the
    # task_id itself, so refuse deletion while that task is still active.
    if prefix in active_task_ids():
        return jsonify({'error': f'Task is still running: {prefix}'}), 409

    root_real = os.path.realpath(_root_path())
    run_dir = os.path.realpath(os.path.join(root_real, prefix))
    # Reject path traversal and symlinks escaping the outputs root.
    if run_dir == root_real or not run_dir.startswith(root_real + os.sep):
        return jsonify({'error': 'Invalid report path'}), 400

    model_report_dir = os.path.realpath(os.path.join(run_dir, OutputsStructure.REPORTS_DIR, model_name))
    if not model_report_dir.startswith(run_dir + os.sep) or not os.path.isdir(model_report_dir):
        return jsonify({'error': f'Report not found: {report_name}'}), 404

    try:
        for sub in (OutputsStructure.REPORTS_DIR, OutputsStructure.PREDICTIONS_DIR, OutputsStructure.REVIEWS_DIR):
            target = os.path.join(run_dir, sub, model_name)
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
        logger.info(f'Deleted eval report {report_name} under {run_dir}')
        return jsonify({'success': True, 'report_name': report_name}), 200
    except Exception as e:
        logger.error(f'Failed to delete report {report_name}: {e}')
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
            'report_list': [_report_to_service_dict(r) for r in report_list],
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
            'report_list': [_report_to_service_dict(r) for r in report_list],
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
            from evalscope.utils.data_utils import get_single_dataset_df
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
        root = os.path.abspath(_root_path())
        prefix, model_name, _ = process_report_name(report_name)
        report_html = os.path.join(root, prefix, OutputsStructure.REPORTS_DIR, 'report.html')

        if not os.path.exists(report_html):
            return jsonify({
                'error': 'Report not yet generated',
                'message': 'The HTML report has not been generated for this evaluation. It may still be in progress.',
            }), 404

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
        chart_type   (str): 'scores' | 'sunburst' | 'dataset_scores' | 'radar' | 'histogram' | 'grouped_bar'
        dataset_name (str): required for chart_type=dataset_scores
        subset_name  (str): required for chart_type=histogram
        theme        (str): 'light' | 'dark'; defaults to the existing dark report theme
    """
    chart_type = request.args.get('chart_type', 'scores')
    root = _root_path()

    try:
        fig = None

        if chart_type == 'radar':
            names_raw = request.args.get('report_names', '')
            names = [n.strip() for n in names_raw.split(';') if n.strip()]
            if not names:
                # Fall back to singular report_name
                single = request.args.get('report_name', '').strip()
                if single:
                    names = [single]
                else:
                    return jsonify({'error': 'report_names or report_name is required for radar'}), 400
            report_list = load_multi_report(root, names)
            quality_df = get_quality_report_df(report_list)
            fig = plot_multi_report_radar(quality_df)
        elif chart_type == 'grouped_bar':
            # Grouped bar chart for multi-model comparison
            names_raw = request.args.get('report_names', '')
            names = [n.strip() for n in names_raw.split(';') if n.strip()]
            if not names:
                return jsonify({'error': 'report_names is required for grouped_bar'}), 400
            report_list = load_multi_report(root, names)
            quality_df = get_quality_report_df(report_list)
            if not quality_df.empty:
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
        elif chart_type == 'histogram':
            # Score distribution histogram from prediction NScore values
            report_name = request.args.get('report_name')
            dataset_name = request.args.get('dataset_name', '')
            subset_name = request.args.get('subset_name', '')
            if not report_name or not dataset_name or not subset_name:
                return jsonify({'error': 'report_name, dataset_name and subset_name are required for histogram'}), 400
            prefix, model_name, _ = process_report_name(report_name)
            work_dir = os.path.join(root, prefix)
            pred_df = get_model_prediction(work_dir, model_name, dataset_name, subset_name)
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
            report_name = request.args.get('report_name')
            if not report_name:
                return jsonify({'error': 'report_name is required'}), 400
            report_list, datasets, _ = load_single_report(root, report_name)
            if chart_type == 'sunburst':
                fig = plot_single_report_sunburst(report_list)
            elif chart_type == 'dataset_scores':
                dataset_name = request.args.get('dataset_name', '')
                if not dataset_name:
                    return jsonify({'error': 'dataset_name is required for dataset_scores'}), 400
                report_df = get_data_frame(report_list=report_list, flatten_metrics=True, flatten_categories=True)
                from evalscope.utils.data_utils import get_single_dataset_df
                ds_df, _ = get_single_dataset_df(report_df, dataset_name)
                fig = plot_single_dataset_scores(get_quality_metric_df(report_list, ds_df))
            else:
                fig = plot_single_report_scores(get_quality_report_df(report_list))

        if fig is None:
            return '<html><body style="background:#0f172a;color:#94a3b8;display:flex;align-items:center;' \
                   'justify-content:center;height:100vh;font-family:sans-serif;">No data to plot</body></html>', \
                   200, {'Content-Type': 'text/html'}

        _apply_chart_theme(fig, request.args.get('theme', 'dark'))
        html = fig.to_html(full_html=True, include_plotlyjs=False, config={'responsive': True})
        plotly_script = f'<script src="{PLOTLY_CDN_URL}" charset="utf-8"></script>'
        html = html.replace('</head>', f'  {plotly_script}\n</head>')
        return html, 200, {'Content-Type': 'text/html'}

    except Exception as e:
        logger.error(f'Failed to generate chart: {e}')
        return jsonify({'error': str(e)}), 500
