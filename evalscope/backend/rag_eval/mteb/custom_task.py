# Copyright (c) Alibaba, Inc. and its affiliates.
"""Custom dataset support for MTEB evaluation.

Provides build_custom_task() factory to create MTEB-compatible task instances
from user-provided local data (JSONL, JSON, CSV, or HuggingFace datasets).
"""
import csv
import json
import os
from collections import defaultdict
from datasets import Dataset, DatasetDict
from mteb import TaskMetadata
from mteb.abstasks import AbsTaskReranking, AbsTaskRetrieval, AbsTaskSTS
from pathlib import Path
from typing import Any, Dict, List, Optional

from evalscope.utils.logger import get_logger

logger = get_logger()

# Default main scores per task type
_DEFAULT_MAIN_SCORES = {
    'Retrieval': 'ndcg_at_10',
    'Reranking': 'map',
    'STS': 'cosine_spearman',
}


def build_custom_task(config: Dict[str, Any]):
    """Factory function to build a custom MTEB task from user configuration.

    Args:
        config: Dict with keys:
            - name (str): Task name identifier
            - type (str): Task type - one of Retrieval, Reranking, STS
            - data_path (str): Path to data directory or file
            - data_format (str): Format of data files (jsonl, json, csv, tsv, hf_dataset)
            - eval_splits (List[str]): Splits to evaluate on, default ["test"]

    Returns:
        An MTEB task instance ready for evaluation.
    """
    task_type = config.get('type', 'Retrieval')
    name = config.get('name', 'custom_task')
    data_path = config.get('data_path')
    data_format = config.get('data_format', 'jsonl')
    eval_splits = config.get('eval_splits', ['test'])

    if not data_path:
        raise ValueError("'data_path' is required for custom tasks.")

    if task_type == 'Retrieval':
        return _build_retrieval_task(name, data_path, data_format, eval_splits)
    elif task_type == 'Reranking':
        return _build_reranking_task(name, data_path, data_format, eval_splits)
    elif task_type == 'STS':
        return _build_sts_task(name, data_path, data_format, eval_splits)
    else:
        raise ValueError(f'Unsupported custom task type: {task_type}. '
                         f'Supported: Retrieval, Reranking, STS')


def _load_file(file_path: str, data_format: str) -> List[Dict[str, Any]]:
    """Load data from a file in the specified format.

    Args:
        file_path: Path to the data file.
        data_format: One of 'jsonl', 'json', 'csv', 'tsv'.

    Returns:
        List of dictionaries representing the records.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f'Data file not found: {file_path}')

    if data_format == 'jsonl':
        records = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records
    elif data_format == 'json':
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
        raise ValueError(f'JSON file must contain a list of objects: {file_path}')
    elif data_format in ('csv', 'tsv'):
        delimiter = '\t' if data_format == 'tsv' else ','
        records = []
        with open(path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f, delimiter=delimiter)
            for row in reader:
                records.append(dict(row))
        return records
    else:
        raise ValueError(f'Unsupported data format: {data_format}. Supported: jsonl, json, csv, tsv')


def _build_metadata(name: str, task_type: str, eval_splits: List[str]) -> TaskMetadata:
    """Build a TaskMetadata instance for a custom task."""
    return TaskMetadata(
        name=name,
        description=f'Custom {task_type} task: {name}',
        reference=None,
        dataset={
            'path': name,
            'revision': 'custom'
        },
        type=task_type,
        category='s2s' if task_type != 'Retrieval' else 's2p',
        modalities=['text'],
        eval_splits=eval_splits,
        eval_langs=['eng-Latn'],
        main_score=_DEFAULT_MAIN_SCORES.get(task_type, 'ndcg_at_10'),
        date=None,
        domains=None,
        task_subtypes=None,
        license=None,
        annotations_creators=None,
        dialect=None,
        sample_creation=None,
        bibtex_citation='',
        descriptive_stats={},
    )


def _build_retrieval_task(
    name: str,
    data_path: str,
    data_format: str,
    eval_splits: List[str],
):
    """Build a custom Retrieval task.

    Expected files in data_path directory:
        - corpus.{format}: {"id": "doc1", "text": "...", "title": "..."}
        - queries.{format}: {"id": "q1", "text": "..."}
        - qrels.{format} or qrels.tsv: {"qid": "q1", "pid": "doc1", "score": 1}
    """
    metadata = _build_metadata(name, 'Retrieval', eval_splits)

    class _CustomRetrieval(AbsTaskRetrieval):
        pass

    _CustomRetrieval.metadata = metadata
    _CustomRetrieval.__name__ = f'CustomRetrieval_{name}'
    _CustomRetrieval.__qualname__ = f'CustomRetrieval_{name}'

    task = _CustomRetrieval()

    # Load data
    data_dir = Path(data_path)
    ext = data_format if data_format in ('jsonl', 'json', 'csv', 'tsv') else 'jsonl'

    corpus_file = _find_file(data_dir, 'corpus', ext)
    queries_file = _find_file(data_dir, 'queries', ext)
    qrels_file = _find_file(data_dir, 'qrels', ext) or _find_file(data_dir, 'qrels', 'tsv')

    if not corpus_file:
        raise FileNotFoundError(f'corpus file not found in {data_path}')
    if not queries_file:
        raise FileNotFoundError(f'queries file not found in {data_path}')
    if not qrels_file:
        raise FileNotFoundError(f'qrels file not found in {data_path}')

    # Determine qrels format
    qrels_ext = Path(qrels_file).suffix.lstrip('.')
    qrels_format = qrels_ext if qrels_ext in ('jsonl', 'json', 'csv', 'tsv') else data_format

    corpus_records = _load_file(corpus_file, data_format)
    queries_records = _load_file(queries_file, data_format)
    qrels_records = _load_file(qrels_file, qrels_format)

    # Build corpus: {doc_id: {"text": text}}
    corpus = {}
    for item in corpus_records:
        doc_id = str(item.get('id', item.get('_id', '')))
        text = item.get('text', '')
        title = item.get('title', '')
        if title:
            text = f'{title} {text}'.strip()
        corpus[doc_id] = {'text': text}

    # Build queries: {qid: text}
    queries = {}
    for item in queries_records:
        qid = str(item.get('id', item.get('_id', '')))
        queries[qid] = item.get('text', '')

    # Build relevant_docs: {qid: {doc_id: score}}
    relevant_docs = defaultdict(dict)
    for item in qrels_records:
        qid = str(item.get('qid', item.get('query-id', '')))
        pid = str(item.get('pid', item.get('corpus-id', item.get('docid', ''))))
        score = int(item.get('score', item.get('label', 1)))
        relevant_docs[qid][pid] = score

    # Assign to all eval splits
    task.corpus = DatasetDict({split: corpus for split in eval_splits})
    task.queries = DatasetDict({split: queries for split in eval_splits})
    task.relevant_docs = DatasetDict({split: dict(relevant_docs) for split in eval_splits})
    task.data_loaded = True

    logger.info(
        f"Custom Retrieval task '{name}' loaded: "
        f'{len(corpus)} docs, {len(queries)} queries, {len(relevant_docs)} qrels'
    )

    return task


def _build_reranking_task(
    name: str,
    data_path: str,
    data_format: str,
    eval_splits: List[str],
):
    """Build a custom Reranking task.

    Expected data format (single file or directory with split files):
        {"query": "...", "positive": ["..."], "negative": ["..."]}
    """
    metadata = _build_metadata(name, 'Reranking', eval_splits)

    class _CustomReranking(AbsTaskReranking):
        pass

    _CustomReranking.metadata = metadata
    _CustomReranking.__name__ = f'CustomReranking_{name}'
    _CustomReranking.__qualname__ = f'CustomReranking_{name}'

    task = _CustomReranking()

    data_path_obj = Path(data_path)
    ext = data_format if data_format in ('jsonl', 'json', 'csv', 'tsv') else 'jsonl'

    dataset_splits = {}
    for split in eval_splits:
        records = _load_split_data(data_path_obj, split, ext, data_format)
        # Convert to MTEB reranking format: query, positive, negative
        samples = []
        for item in records:
            query = item.get('query', '')
            positive = item.get('positive', [])
            negative = item.get('negative', [])
            if isinstance(positive, str):
                positive = [positive]
            if isinstance(negative, str):
                negative = [negative]
            samples.append({
                'query': query,
                'positive': positive,
                'negative': negative,
            })
        dataset_splits[split] = Dataset.from_list(samples)

    task.dataset = DatasetDict(dataset_splits)
    task.dataset_transform()
    task.data_loaded = True

    logger.info(
        f"Custom Reranking task '{name}' loaded: "
        f'{sum(len(ds) for ds in dataset_splits.values())} samples across {len(eval_splits)} splits'
    )

    return task


def _build_sts_task(
    name: str,
    data_path: str,
    data_format: str,
    eval_splits: List[str],
):
    """Build a custom STS task.

    Expected data format (single file or directory with split files):
        {"sentence1": "...", "sentence2": "...", "score": 0.8}
    """
    metadata = _build_metadata(name, 'STS', eval_splits)

    class _CustomSTS(AbsTaskSTS):

        @property
        def metadata_dict(self) -> dict:
            md = super().metadata_dict
            md['min_score'] = 0
            md['max_score'] = 1
            return md

    _CustomSTS.metadata = metadata
    _CustomSTS.__name__ = f'CustomSTS_{name}'
    _CustomSTS.__qualname__ = f'CustomSTS_{name}'

    task = _CustomSTS()

    data_path_obj = Path(data_path)
    ext = data_format if data_format in ('jsonl', 'json', 'csv', 'tsv') else 'jsonl'

    dataset_splits = {}
    for split in eval_splits:
        records = _load_split_data(data_path_obj, split, ext, data_format)
        samples = []
        for item in records:
            sentence1 = item.get('sentence1', '')
            sentence2 = item.get('sentence2', '')
            score = float(item.get('score', 0.0))
            samples.append({
                'sentence1': sentence1,
                'sentence2': sentence2,
                'score': score,
            })
        dataset_splits[split] = Dataset.from_list(samples)

    task.dataset = DatasetDict(dataset_splits)
    task.dataset_transform()
    task.data_loaded = True

    logger.info(
        f"Custom STS task '{name}' loaded: "
        f'{sum(len(ds) for ds in dataset_splits.values())} samples across {len(eval_splits)} splits'
    )

    return task


def _find_file(data_dir: Path, prefix: str, ext: str) -> Optional[str]:
    """Find a data file by prefix and extension in a directory.

    Tries: {prefix}.{ext}, then falls back to any file starting with prefix.
    """
    # Try exact match first
    candidate = data_dir / f'{prefix}.{ext}'
    if candidate.exists():
        return str(candidate)

    # Try common extensions
    for try_ext in ('jsonl', 'json', 'csv', 'tsv'):
        candidate = data_dir / f'{prefix}.{try_ext}'
        if candidate.exists():
            return str(candidate)

    return None


def _load_split_data(
    data_path: Path,
    split: str,
    ext: str,
    data_format: str,
) -> List[Dict[str, Any]]:
    """Load data for a specific split.

    Resolution order:
        1. data_path is a file → load it directly (all splits use same file)
        2. data_path is a directory → look for {split}.{ext} or data.{ext}
    """
    if data_path.is_file():
        return _load_file(str(data_path), data_format)

    # Directory: look for split-specific file
    split_file = data_path / f'{split}.{ext}'
    if split_file.exists():
        return _load_file(str(split_file), data_format)

    # Fallback to data.{ext}
    data_file = data_path / f'data.{ext}'
    if data_file.exists():
        return _load_file(str(data_file), data_format)

    # Try any extension for the split name
    for try_ext in ('jsonl', 'json', 'csv', 'tsv'):
        candidate = data_path / f'{split}.{try_ext}'
        if candidate.exists():
            return _load_file(str(candidate), try_ext)

    raise FileNotFoundError(
        f"No data file found for split '{split}' in {data_path}. "
        f'Expected: {split}.{ext} or data.{ext}'
    )
