# Copyright (c) Alibaba, Inc. and its affiliates.
"""ModelScope data loader for MTEB tasks.

Patches MTEB task ``load_data`` methods so that datasets are fetched from
ModelScope instead of HuggingFace. When ``hub`` is anything other than
``modelscope``, the patching is a no-op and MTEB loads natively.
"""
import types
from collections import defaultdict
from datasets import DatasetDict
from typing import List, Optional

from evalscope.utils.logger import get_logger

logger = get_logger()


def patch_tasks_for_modelscope(tasks: List, limits: Optional[int] = None) -> None:
    """Patch a list of MTEB tasks to load data from ModelScope.

    Args:
        tasks: List of MTEB task instances (subclasses of ``mteb.AbsTask``).
        limits: Optional cap on the number of samples per split. For Retrieval
            tasks this caps the number of queries; for other task types it
            caps the number of rows per split.
    """
    for task in tasks:
        _patch_single_task(task, limits=limits)


def _is_retrieval_task(task) -> bool:
    """Check if a task requires retrieval-style data loading.

    In MTEB 2.x, Reranking tasks inherit from AbsTaskRetrieval and expect
    corpus/queries/relevant_docs format, so we check both the metadata type
    and the class hierarchy.
    """
    task_type = getattr(task.metadata, 'type', None)
    if task_type in ('Retrieval', 'Reranking'):
        return True
    try:
        from mteb.abstasks.retrieval import AbsTaskRetrieval
        return isinstance(task, AbsTaskRetrieval)
    except ImportError:
        return False


def _patch_single_task(task, limits: Optional[int] = None) -> None:
    """Patch a single MTEB task's ``load_data`` method for ModelScope loading."""
    if _is_retrieval_task(task):

        def ms_load_retrieval(self, **kwargs):
            if getattr(self, 'data_loaded', False):
                return
            _load_retrieval_from_modelscope(self, limits=limits)

        task.load_data = types.MethodType(ms_load_retrieval, task)
    else:

        def ms_load_data(self, **kwargs):
            if getattr(self, 'data_loaded', False):
                return
            _load_generic_from_modelscope(self, limits=limits)

        task.load_data = types.MethodType(ms_load_data, task)


# Special naming conventions on ModelScope C-MTEB organization
_MODELSCOPE_NAME_MAP = {
    'MMarcoReranking': 'Mmarco-reranking',
    'CMedQAv1': 'CMedQAv1-reranking',
    'CMedQAv2': 'CMedQAv2-reranking',
    'TNews': 'TNews-classification',
    'IFlyTek': 'IFlyTek-classification',
    'Ocnli': 'OCNLI',
}


def _get_modelscope_path(path: str) -> str:
    """Convert HuggingFace dataset path to ModelScope path.

    Rules:
        - ``mteb/xxx`` → ``C-MTEB/xxx`` (with special name mapping for some datasets)
        - Other paths left unchanged
    """
    if not path.startswith('mteb/'):
        return path
    dataset_name = path[len('mteb/'):]
    # Apply special naming if needed
    mapped_name = _MODELSCOPE_NAME_MAP.get(dataset_name, dataset_name)
    return f'C-MTEB/{mapped_name}'


def _load_generic_from_modelscope(task, limits: Optional[int] = None) -> None:
    """Load a generic (non-retrieval) MTEB dataset from ModelScope."""
    from modelscope import MsDataset

    path = task.metadata.dataset.get('path', '')
    ms_path = _get_modelscope_path(path)
    task_name = getattr(task.metadata, 'name', '<unknown>')

    logger.info(f"Loading dataset '{ms_path}' from ModelScope for task '{task_name}'")

    try:
        dataset = MsDataset.load(ms_path)
    except Exception as e:
        logger.warning(f'Failed to load from ModelScope ({ms_path}), falling back to HuggingFace native loading: {e}')
        # Fallback: let MTEB handle it natively.
        task.data_loaded = False
        task.__class__.load_data(task)
        return

    if limits is not None:
        try:
            dataset = DatasetDict({split: ds.select(range(min(limits, len(ds)))) for split, ds in dataset.items()})
        except Exception as e:
            logger.warning(f"Failed to apply limits={limits} to dataset '{ms_path}': {e}")

    task.dataset = dataset
    task.dataset_transform()
    task.data_loaded = True


def _get_val(obj, key):
    """Get value from dict or object attribute."""
    return obj.get(key) if isinstance(obj, dict) else getattr(obj, key, None)


def _set_val(obj, key, val):
    """Set value on dict or object attribute."""
    if isinstance(obj, dict):
        obj[key] = val
    else:
        setattr(obj, key, val)


def _apply_retrieval_limits_after_native_load(task, limits=None):
    """Best-effort application of query limits after MTEB native data loading.

    Handles both nested (subset -> split -> data) and flat (split -> data)
    DatasetDict structures, as well as dict-based and object-based split data.
    """
    if limits is None or not getattr(task, 'data_loaded', False):
        return

    if not hasattr(task, 'dataset') or not task.dataset:
        return

    try:
        # Normalize: detect flat structure (first value has 'queries')
        dataset_dict = task.dataset
        first_val = next(iter(dataset_dict.values()), None)
        if first_val is not None and ((isinstance(first_val, dict) and 'queries' in first_val)
                                      or hasattr(first_val, 'queries')):
            dataset_dict = {'default': dataset_dict}

        for subset_key, subset_data in dataset_dict.items():
            if not hasattr(subset_data, 'items'):
                continue
            for split_key, split_data in subset_data.items():
                queries = _get_val(split_data, 'queries')
                if queries is None or len(queries) <= limits:
                    continue

                # Get query IDs to keep
                limited_queries = queries.select(range(limits))
                keep_qids = set(str(q) for q in limited_queries['id'])

                _set_val(split_data, 'queries', limited_queries)
                logger.info(
                    f"Applied limits={limits} to queries in '{subset_key}/{split_key}' "
                    f'(was {len(queries)}, now {len(limited_queries)}).'
                )

                # Filter relevant_docs to keep only limited query IDs
                relevant_docs = _get_val(split_data, 'relevant_docs')
                if relevant_docs is not None:
                    try:
                        if isinstance(relevant_docs, dict):
                            _set_val(
                                split_data, 'relevant_docs', {
                                    k: v
                                    for k, v in relevant_docs.items()
                                    if k in keep_qids
                                }
                            )
                        elif hasattr(relevant_docs, 'filter'):
                            _set_val(
                                split_data, 'relevant_docs',
                                relevant_docs.filter(lambda x: str(x.get('query-id', x.get('qid', ''))) in keep_qids)
                            )
                    except Exception as e:
                        logger.warning(f'Could not filter relevant_docs: {e}')

                # Filter top_ranked to keep only limited query IDs
                top_ranked = _get_val(split_data, 'top_ranked')
                if top_ranked is not None:
                    try:
                        if isinstance(top_ranked, dict):
                            _set_val(split_data, 'top_ranked', {k: v for k, v in top_ranked.items() if k in keep_qids})
                        elif hasattr(top_ranked, 'filter'):
                            _set_val(
                                split_data, 'top_ranked',
                                top_ranked.filter(lambda x: str(x.get('query-id', x.get('qid', ''))) in keep_qids)
                            )
                    except Exception as e:
                        logger.warning(f'Could not filter top_ranked: {e}')
    except Exception as e:
        logger.warning(f'Failed to apply limits after native loading: {e}')


def _load_retrieval_from_modelscope(task, limits: Optional[int] = None) -> None:
    """Load a Retrieval MTEB dataset from ModelScope.

    Retrieval datasets in MTEB consist of corpus, queries, and qrels (relevant
    documents). On ModelScope, qrels live in a separate dataset suffixed with
    ``-qrels``.
    """
    from modelscope import MsDataset

    path = task.metadata.dataset.get('path', '')
    ms_path = _get_modelscope_path(path)
    task_name = getattr(task.metadata, 'name', '<unknown>')
    eval_splits = getattr(task.metadata, 'eval_splits', None) or ['test']
    eval_split = eval_splits[0]

    logger.info(f"Loading retrieval dataset '{ms_path}' from ModelScope for task '{task_name}'")

    try:
        dataset = MsDataset.load(ms_path)
        qrels_dataset = MsDataset.load(ms_path + '-qrels')
    except Exception as e:
        logger.warning(
            f'Failed to load retrieval data from ModelScope ({ms_path}), '
            f'falling back to HuggingFace native loading: {e}'
        )
        task.data_loaded = False
        task.__class__.load_data(task)
        _apply_retrieval_limits_after_native_load(task, limits=limits)
        return

    # Build corpus: {doc_id: {"text": text}}
    corpus = {}
    if 'corpus' in dataset:
        for item in dataset['corpus']:
            doc_id = str(item.get('id', item.get('_id', '')))
            text = item.get('text', '')
            title = item.get('title', '')
            if title:
                text = f'{title} {text}'.strip()
            corpus[doc_id] = {'text': text}

    # Build queries: {qid: text}
    queries = {}
    if 'queries' in dataset:
        for item in dataset['queries']:
            qid = str(item.get('id', item.get('_id', '')))
            queries[qid] = item.get('text', '')

    # Build relevant_docs: {qid: {doc_id: score}}
    relevant_docs = defaultdict(dict)
    if eval_split in qrels_dataset:
        for item in qrels_dataset[eval_split]:
            qid = str(item.get('qid', item.get('query-id', '')))
            pid = str(item.get('pid', item.get('corpus-id', '')))
            score = int(item.get('score', 1))
            relevant_docs[qid][pid] = score

    # Apply limits to queries (and matching qrels).
    if limits is not None and len(queries) > limits:
        limited_qids = list(queries.keys())[:limits]
        queries = {qid: queries[qid] for qid in limited_qids}
        relevant_docs = defaultdict(dict, {qid: relevant_docs[qid] for qid in limited_qids if qid in relevant_docs})

    task.corpus = DatasetDict({eval_split: corpus})
    task.queries = DatasetDict({eval_split: queries})
    task.relevant_docs = DatasetDict({eval_split: dict(relevant_docs)})
    task.data_loaded = True
