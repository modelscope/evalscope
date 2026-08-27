"""
Data loading and processing utilities for reports and predictions.
"""

import glob
import os
from typing import Any, Dict, List, Union

import pandas as pd
from pydantic import ValidationError

from evalscope.api.evaluator import CacheManager, ReviewResult
from evalscope.constants import DataCollection
from evalscope.metrics.semantics import format_metric_value
from evalscope.metrics.semantics.ranking import bounded_quality_ratio
from evalscope.report import Report, ReportKey, ReportRef, get_data_frame, get_report_list
from evalscope.utils.io_utils import OutputsStructure, jsonl_to_list, yaml_to_dict
from evalscope.utils.logger import get_logger

logger = get_logger()


def scan_report_refs(root_path: str) -> List[ReportRef]:
    """Scan an outputs root for every model report it holds.

    Returns:
        List[ReportRef]: One reference per ``<run_id>/reports/<model_id>`` directory, newest first.
            Directories whose names cannot form a reference (e.g. a manually created nested path)
            are skipped with a warning rather than breaking the whole listing.
    """
    logger.debug(f'Scanning for report folders in {root_path}')
    if not os.path.exists(root_path):
        return []

    refs: List[ReportRef] = []
    for run_dir in glob.glob(os.path.join(root_path, '*')):
        reports_path = os.path.join(run_dir, OutputsStructure.REPORTS_DIR)
        if not os.path.exists(reports_path):
            continue

        for model_dir in glob.glob(os.path.join(reports_path, '*')):
            if not os.path.isdir(model_dir):
                continue
            try:
                refs.append(ReportRef(run_id=os.path.basename(run_dir), model_id=os.path.basename(model_dir)))
            except ValidationError as e:
                logger.warning(f'Skipping report directory {model_dir}: {e}')

    refs.sort(key=lambda ref: ref.key, reverse=True)
    logger.debug(f'reports: {[ref.key for ref in refs]}')
    return refs


def report_model_dir(root_path: str, ref: ReportRef) -> str:
    """Directory holding the per-dataset report files of one model report."""
    return os.path.join(root_path, ref.run_id, OutputsStructure.REPORTS_DIR, ref.model_id)


def load_report_bundle(root_path: str, ref: ReportRef) -> tuple[List[Report], List[str], Dict[str, Any]]:
    """Load one model report together with its datasets and the run's task configuration.

    The dataset list is derived from the loaded reports rather than from the identifier, so it always
    describes what is actually on disk.
    """
    model_dir = report_model_dir(root_path, ref)
    report_list = get_report_list([model_dir])
    if not report_list:
        raise FileNotFoundError(f'No report files found in {model_dir}')
    datasets = list(dict.fromkeys(report.dataset_name for report in report_list))

    configs_dir = os.path.join(root_path, ref.run_id, OutputsStructure.CONFIGS_DIR)
    config_files = glob.glob(os.path.join(configs_dir, '*.yaml'))
    if not config_files:
        raise FileNotFoundError(f'No configuration files found in {configs_dir}')
    task_cfg = yaml_to_dict(config_files[0])
    return report_list, datasets, task_cfg


def load_multi_report_groups(root_path: str, refs: List[ReportRef]) -> List[tuple[ReportRef, List[Report]]]:
    """Load reports while retaining the reference that owns each group."""
    return [(ref, get_report_list([report_model_dir(root_path, ref)])) for ref in refs]


def get_acc_report_df(report_list: List[Report]) -> pd.DataFrame:
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
            # `primary_metric` is the declared `role=primary` metric, or the inferred headline
            # when the benchmark declared none (see `Report._find_primary_metric`). It is only
            # `None` for a report with no metric at all, which then shows no score.
            primary_metric = report.primary_metric
            item = {
                ReportKey.model_name: report.model_name,
                ReportKey.dataset_name: report.dataset_name,
                ReportKey.score: primary_metric.score if primary_metric else None,
                ReportKey.num: primary_metric.num if primary_metric else 0,
            }
            data_dict.append(item)
    df = pd.DataFrame.from_dict(data_dict, orient='columns')

    return df


def get_quality_report_df(report_list: List[Report]) -> pd.DataFrame:
    """Build chart rows on a comparable 0-1 quality axis.

    Unbounded and diagnostic metrics are omitted because no honest normalization exists for them.
    Their native score is never replaced in the report itself; the formatted value travels next
    to the quality ratio for chart labels and tooltips.
    """
    rows = []
    for report in report_list:
        metric = report.primary_metric
        if metric is None or metric.semantics is None:
            continue
        quality_ratio = bounded_quality_ratio(metric.score, metric.semantics)
        if quality_ratio is None:
            continue
        rows.append(
            {
                ReportKey.model_name: report.model_name,
                ReportKey.dataset_name: report.dataset_name,
                ReportKey.metric_name: metric.name,
                ReportKey.score: quality_ratio,
                ReportKey.raw_score: metric.score,
                ReportKey.display_score: format_metric_value(metric.score, metric.semantics),
                ReportKey.num: metric.num,
            }
        )
    return pd.DataFrame.from_records(
        rows,
        columns=[
            ReportKey.model_name,
            ReportKey.dataset_name,
            ReportKey.metric_name,
            ReportKey.score,
            ReportKey.raw_score,
            ReportKey.display_score,
            ReportKey.num,
        ],
    )


def get_comparison_quality_report_df(report_groups: List[tuple[ReportRef, List[Report]]]) -> pd.DataFrame:
    """Build comparison chart rows without collapsing separate runs of the same model."""
    model_counts: Dict[str, int] = {}
    for ref, _ in report_groups:
        model_counts[ref.model_id] = model_counts.get(ref.model_id, 0) + 1

    frames = []
    for ref, reports in report_groups:
        frame = get_quality_report_df(reports)
        if frame.empty:
            continue
        display_name = f'{ref.model_id} ({ref.run_id})' if model_counts[ref.model_id] > 1 else ref.model_id
        frame[ReportKey.model_name] = display_name
        frames.append(frame)

    return pd.concat(frames, ignore_index=True) if frames else get_quality_report_df([])


def get_quality_metric_df(report_list: List[Report], metric_df: pd.DataFrame) -> pd.DataFrame:
    """Normalize metric rows for charts while retaining their native display value."""
    semantics_by_metric = {
        (report.model_name, report.dataset_name, metric.name): metric.semantics
        for report in report_list
        for metric in report.metrics
    }
    rows = []
    for _, row in metric_df.iterrows():
        semantics = semantics_by_metric.get(
            (row[ReportKey.model_name], row[ReportKey.dataset_name], row[ReportKey.metric_name])
        )
        raw_score = row[ReportKey.score]
        quality_ratio = bounded_quality_ratio(raw_score, semantics)
        if quality_ratio is None:
            continue
        item = row.to_dict()
        item[ReportKey.raw_score] = raw_score
        item[ReportKey.display_score] = format_metric_value(raw_score, semantics)
        item[ReportKey.score] = quality_ratio
        rows.append(item)

    columns = list(metric_df.columns)
    for column in (ReportKey.raw_score, ReportKey.display_score):
        if column not in columns:
            columns.append(column)
    return pd.DataFrame.from_records(rows, columns=columns)


def get_compare_report_df(acc_df: pd.DataFrame) -> pd.DataFrame:
    df = acc_df.pivot_table(index=ReportKey.model_name, columns=ReportKey.dataset_name, values=ReportKey.score)
    df.reset_index(inplace=True)

    return df


def get_single_dataset_df(df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    return df[df[ReportKey.dataset_name] == dataset_name]


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


# Serialised ContentBlock types that may carry a server-side media path.  The
# payload field of each of these blocks is named after the block type.
_MEDIA_BLOCK_TYPES = frozenset({'image', 'audio', 'video'})


def _absolutize_media_path(block: Dict[str, Any]) -> Dict[str, Any]:
    """Rewrite a local media path in a serialised ContentBlock to an absolute path.

    A renderer can only resolve a server-side file when it is given an absolute
    path: a relative path is indistinguishable from a base64 payload on the
    client side.  This mirrors what :func:`messages_to_markdown` does for the
    markdown chain, so both chains expose local media the same way.

    Args:
        block (Dict[str, Any]): A serialised ContentBlock, mutated in place.

    Returns:
        Dict[str, Any]: The same block, for convenient chaining.
    """
    block_type = block.get('type')
    if block_type not in _MEDIA_BLOCK_TYPES:
        return block
    value = block.get(block_type)
    if isinstance(value, str) and value and not value.startswith('data:') and os.path.isfile(value):
        block[block_type] = os.path.abspath(value)
    return block


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
                serialised_content = [_absolutize_media_path(c.model_dump()) for c in m.content]
            else:
                # Text-only / legacy path – content is already a str.
                serialised_content = m.content

            entry: Dict[str, Any] = {
                'id': m.id,
                'role': m.role,
                'content': serialised_content,
                'source': m.source,
                'metadata': m.metadata,
                'perf_metrics': m.perf_metrics.model_dump() if m.perf_metrics else None,
            }

            tool_call_id = getattr(m, 'tool_call_id', None)
            if tool_call_id:
                entry['tool_call_id'] = tool_call_id

            # Assistant: tool_calls + model name (if any)
            if m.role == 'assistant':
                tool_calls = getattr(m, 'tool_calls', None)
                if tool_calls:
                    entry['tool_calls'] = [
                        {
                            'id': tc.id,
                            'function': tc.function.name,
                            'arguments': tc.function.arguments,
                        }
                        for tc in tool_calls
                    ]
                model_name = getattr(m, 'model', None)
                if model_name:
                    entry['model'] = model_name

            # Tool: function name + tool_call_id + error
            if m.role == 'tool':
                function = getattr(m, 'function', None)
                if function:
                    entry['function'] = function
                error = getattr(m, 'error', None)
                if error:
                    entry['error'] = {
                        'type': getattr(error, 'type', None),
                        'message': getattr(error, 'message', ''),
                    }

            messages_data.append(entry)
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
            messages_data.append(
                {
                    'role': 'assistant',
                    'content': prediction,
                    'perf_metrics': fallback_perf,
                }
            )

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
    main_value = score.main_value

    return {
        'Index': str(review_result.index),
        'Input': review_result.messages_markdown.replace('\n', '\n\n'),  # for markdown
        'Metadata': sample_score.sample_metadata,
        'Generated': prediction or '',  # Ensure no None value
        'Gold': review_result.target or '*No Gold Provided*',
        'Pred': (extracted_prediction if extracted_prediction != prediction else '*Same as Generated*')
        or '',  # Ensure no None value
        'Score': score.model_dump(exclude_none=True),
        # ``None`` when the sample carries no usable value: a judge that could not be
        # scored is not a sample that scored 0.
        'NScore': normalize_score(main_value) if main_value is not None else None,
        'Status': score.status.value,
        'PerfMetrics': fallback_perf,
        'Messages': messages_data,
        'AgentTrace': review_result.agent_trace.model_dump(exclude_none=True) if review_result.agent_trace else None,
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
        review_result = ReviewResult.from_cache_item(cache)
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
