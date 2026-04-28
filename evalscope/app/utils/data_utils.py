"""
Data loading and processing utilities for the Evalscope dashboard.
"""
import glob
import os
import pandas as pd
from typing import Any, Dict, List, Union

from evalscope.api.evaluator import CacheManager, ReviewResult
from evalscope.constants import DataCollection
from evalscope.report import Report, ReportKey, get_data_frame, get_report_list
from evalscope.utils.io_utils import OutputsStructure, jsonl_to_list, yaml_to_dict
from evalscope.utils.logger import get_logger
from ..constants import DATASET_TOKEN, MODEL_TOKEN, REPORT_TOKEN

logger = get_logger()


def scan_for_report_folders(root_path):
    """Scan for folders containing reports subdirectories"""
    logger.debug(f'Scanning for report folders in {root_path}')
    if not os.path.exists(root_path):
        return []

    reports = []
    # Iterate over all folders in the root path
    for folder in glob.glob(os.path.join(root_path, '*')):
        # Check if reports folder exists
        reports_path = os.path.join(folder, OutputsStructure.REPORTS_DIR)
        if not os.path.exists(reports_path):
            continue

        # Iterate over all items in reports folder
        for model_item in glob.glob(os.path.join(reports_path, '*')):
            if not os.path.isdir(model_item):
                continue
            datasets = []
            for dataset_item in glob.glob(os.path.join(model_item, '*.json')):
                base_name = os.path.basename(dataset_item)
                if base_name == DataCollection.REPORT_NAME:
                    continue

                datasets.append(os.path.splitext(base_name)[0])
            datasets = DATASET_TOKEN.join(datasets)
            reports.append(
                f'{os.path.basename(folder)}{REPORT_TOKEN}{os.path.basename(model_item)}{MODEL_TOKEN}{datasets}'
            )

    reports = sorted(reports, reverse=True)
    logger.debug(f'reports: {reports}')
    return reports


def process_report_name(report_name: str):
    prefix, report_name = report_name.split(REPORT_TOKEN)
    model_name, datasets = report_name.split(MODEL_TOKEN)
    datasets = datasets.split(DATASET_TOKEN)
    return prefix, model_name, datasets


def load_single_report(root_path: str, report_name: str):
    prefix, model_name, datasets = process_report_name(report_name)
    report_path_list = os.path.join(root_path, prefix, OutputsStructure.REPORTS_DIR, model_name)
    report_list = get_report_list([report_path_list])

    config_files = glob.glob(os.path.join(root_path, prefix, OutputsStructure.CONFIGS_DIR, '*.yaml'))
    if not config_files:
        raise FileNotFoundError(
            f'No configuration files found in {os.path.join(root_path, prefix, OutputsStructure.CONFIGS_DIR)}'
        )
    task_cfg_path = config_files[0]
    task_cfg = yaml_to_dict(task_cfg_path)
    return report_list, datasets, task_cfg


def load_multi_report(root_path: str, report_names: List[str]):
    report_list = []
    for report_name in report_names:
        prefix, model_name, datasets = process_report_name(report_name)
        report_path_list = os.path.join(root_path, prefix, OutputsStructure.REPORTS_DIR, model_name)
        reports = get_report_list([report_path_list])
        report_list.extend(reports)
    return report_list


def get_acc_report_df(report_list: List[Report]):
    data_dict = []
    for report in report_list:
        if report.name == DataCollection.NAME:
            for metric in report.metrics:
                for category in metric.categories:
                    item = {
                        ReportKey.model_name: report.model_name,
                        ReportKey.dataset_name: '/'.join(category.name),
                        ReportKey.score: category.score,
                        ReportKey.num: category.num,
                    }
                    data_dict.append(item)
        else:
            item = {
                ReportKey.model_name: report.model_name,
                ReportKey.dataset_name: report.dataset_name,
                ReportKey.score: report.score,
                ReportKey.num: report.metrics[0].num,
            }
            data_dict.append(item)
    df = pd.DataFrame.from_dict(data_dict, orient='columns')

    styler = style_df(df, columns=[ReportKey.score])
    return df, styler


def style_df(df: pd.DataFrame, columns: List[str] = None):
    # Apply background gradient to the specified columns
    styler = df.style.background_gradient(subset=columns, cmap='RdYlGn', vmin=0.0, vmax=1.0, axis=0)
    # Format the dataframe with a precision of 4 decimal places
    styler.format(precision=4)
    return styler


def get_compare_report_df(acc_df: pd.DataFrame):
    df = acc_df.pivot_table(index=ReportKey.model_name, columns=ReportKey.dataset_name, values=ReportKey.score)
    df.reset_index(inplace=True)

    styler = style_df(df)
    return df, styler


def get_single_dataset_df(df: pd.DataFrame, dataset_name: str):
    df = df[df[ReportKey.dataset_name] == dataset_name]
    styler = style_df(df, columns=[ReportKey.score])
    return df, styler


def get_report_analysis(report_list: List[Report], dataset_name: str) -> str:
    for report in report_list:
        if report.dataset_name == dataset_name:
            return report.analysis
    return 'N/A'


def _load_perf_map(cache_manager: CacheManager, dataset_name: str, subset_name: str) -> Dict[int, Any]:
    """Build an index -> perf_metrics mapping from the prediction cache.

    Old single-turn caches store perf data at the ModelOutput level rather than
    on individual ChatMessages.  This map is used as a fallback so that the
    PerfChip in the UI still renders correctly for legacy cache files.

    Returns an empty dict when the prediction cache is absent or cannot be read.
    """
    perf_map: Dict[int, Any] = {}
    try:
        pred_subset = 'default' if dataset_name == DataCollection.NAME else subset_name
        pred_cache_path = cache_manager.get_prediction_cache_path(pred_subset)
        if os.path.exists(pred_cache_path):
            for item in jsonl_to_list(pred_cache_path):
                idx = item.get('index')
                mo = item.get('model_output') or {}
                pm = mo.get('perf_metrics')
                if pm is not None and idx is not None:
                    # Extract only the fields we want to expose
                    perf_map[int(idx)] = {
                        'latency': pm.get('latency'),
                        'ttft': pm.get('ttft'),
                        'tpot': pm.get('tpot'),
                        'input_tokens': pm.get('input_tokens'),
                        'output_tokens': pm.get('output_tokens'),
                    }
    except Exception as e:
        logger.debug(f'Could not load perf metrics from prediction cache: {e}')
    return perf_map


def _serialize_messages(review_result: ReviewResult) -> List[Dict[str, Any]]:
    """Serialize a ReviewResult's message list into frontend-compatible dicts.

    Each entry follows the ChatMessage wire format expected by ``types.ts``:

    .. code-block:: text

        { role, content, perf_metrics? }

    ``content`` is either a plain string (text-only / legacy) or a list of
    ContentBlock dicts for multimodal messages.  Block types include:

    - ``{type:'text',  text:'...'}``          – markdown text
    - ``{type:'reasoning', reasoning:'...'}`` – chain-of-thought block
    - ``{type:'image', image:'<url|b64>', detail:'auto'}``
    - ``{type:'audio', audio:'<url|b64>', format:'mp3'}``
    - ``{type:'video', video:'<url|b64>', format:'mp4'}``
    - ``{type:'data',  data:{...}}``           – opaque provider payload

    Returns an empty list on serialisation failure (error is logged at DEBUG).
    """
    messages_data = []
    try:
        for m in review_result.messages:
            if isinstance(m.content, list):
                # Multimodal path – preserve all block types (image, audio,
                # reasoning, text, …).  model_dump() mirrors the Python
                # ContentBase subclasses to plain dicts that match the
                # TypeScript ContentBlock interface in types.ts.
                serialised_content = [c.model_dump() for c in m.content]
            else:
                # Text-only / legacy path – content is already a str.
                serialised_content = m.content

            messages_data.append({
                'id': m.id,
                'role': m.role,
                'content': serialised_content,
                'perf_metrics': m.perf_metrics.model_dump() if m.perf_metrics else None,
            })
    except Exception as e:
        logger.debug(f'Could not serialize messages for prediction row: {e}')
        return []
    return messages_data


def _apply_legacy_perf_compat(
    messages_data: List[Dict[str, Any]],
    fallback_perf: Any,
    prediction: Any,
) -> List[Dict[str, Any]]:
    """Apply legacy-compatibility fixes for perf_metrics on message lists.

    Two scenarios are handled:

    1. **Back-fill** – the messages list exists but the last assistant message
       has no ``perf_metrics`` (legacy cache format).  The fallback value from
       the prediction-level cache is propagated so the UI shows consistent
       performance chips.

    2. **Inject** – older review caches stored the model output only in
       ``score.prediction`` and did NOT include an assistant ChatMessage in
       ``review_result.messages``.  An assistant message is appended when
       absent to avoid a blank conversation view in the UI.

    Returns the (possibly mutated) messages list.
    """
    # Back-fill perf onto the last assistant message when it is missing
    if messages_data and fallback_perf:
        for msg in reversed(messages_data):
            if msg['role'] == 'assistant' and msg['perf_metrics'] is None:
                msg['perf_metrics'] = fallback_perf
                break

    # Inject a synthetic assistant message for legacy caches that omit it
    if prediction:
        has_assistant = any(m['role'] == 'assistant' for m in messages_data)
        if not has_assistant:
            messages_data.append({
                'role': 'assistant',
                'content': prediction,
                'perf_metrics': fallback_perf,
            })

    return messages_data


def _build_prediction_row(
    review_result: ReviewResult,
    fallback_perf: Any,
    messages_data: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Assemble a single prediction row dict from a ReviewResult.

    The returned dict is used directly as a DataFrame row in
    :func:`get_model_prediction`.
    """
    sample_score = review_result.sample_score
    score = sample_score.score

    prediction = score.prediction
    extracted_prediction = score.extracted_prediction

    return {
        'Index': str(review_result.index),
        'Input': review_result.messages_markdown.replace('\n', '\n\n'),  # for markdown
        'Metadata': sample_score.sample_metadata,
        'Generated': prediction or '',  # Ensure no None value
        'Gold': review_result.target or '*No Gold Provided*',
        'Pred': (extracted_prediction if extracted_prediction != prediction else '*Same as Generated*')
        or '',  # Ensure no None value
        'Score': score.model_dump(exclude_none=True),
        'NScore': normalize_score(score.main_value),
        'PerfMetrics': fallback_perf,
        'Messages': messages_data,
    }


def get_model_prediction(work_dir: str, model_name: str, dataset_name: str, subset_name: str):
    """Load all prediction / review rows for a given model + dataset subset.

    Returns a :class:`pandas.DataFrame` where every row corresponds to one
    evaluated sample.  Columns match the keys produced by
    :func:`_build_prediction_row`.
    """
    # Locate and load the review cache for this model / dataset / subset
    outputs = OutputsStructure(work_dir, is_make=False)
    cache_manager = CacheManager(outputs, model_name, dataset_name)
    cache_key = 'default' if dataset_name == DataCollection.NAME else subset_name
    review_cache_path = cache_manager.get_review_cache_path(cache_key)
    logger.debug(f'review_path: {review_cache_path}')
    review_caches = jsonl_to_list(review_cache_path)

    # Build index -> perf_metrics fallback map from the prediction cache
    perf_map = _load_perf_map(cache_manager, dataset_name, subset_name)

    ds = []
    for cache in review_caches:
        review_result = ReviewResult.model_validate(cache)
        sample_score = review_result.sample_score

        # For DataCollection, filter to the requested subset
        if dataset_name == DataCollection.NAME:
            collection_info = sample_score.sample_metadata[DataCollection.INFO]
            sample_dataset_name = collection_info.get('dataset_name', 'default')
            sample_subset_name = collection_info.get('subset_name', 'default')
            if f'{sample_dataset_name}/{sample_subset_name}' != subset_name:
                continue

        # Serialise messages to frontend-compatible dicts
        messages_data = _serialize_messages(review_result)

        # Resolve per-sample fallback perf from the prediction-level cache
        fallback_perf = perf_map.get(int(review_result.index))

        # Apply legacy-compatibility fixes for perf and missing assistant turns
        prediction = sample_score.score.prediction
        messages_data = _apply_legacy_perf_compat(messages_data, fallback_perf, prediction)

        ds.append(_build_prediction_row(review_result, fallback_perf, messages_data))

    return pd.DataFrame(ds)


def normalize_score(score):
    try:
        if isinstance(score, bool):
            return 1.0 if score else 0.0
        elif isinstance(score, dict):
            for key in score:
                return float(score[key])
            return 0.0
        else:
            return float(score)
    except (ValueError, TypeError):
        return 0.0
