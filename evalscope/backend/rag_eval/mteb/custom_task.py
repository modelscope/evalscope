# Copyright (c) Alibaba, Inc. and its affiliates.
"""Custom dataset support for MTEB evaluation.

Provides build_custom_task() factory to create MTEB-compatible Retrieval task
instances from user-provided local data in JSONL format.
"""
import json
from collections import defaultdict
from datasets import DatasetDict
from mteb import TaskMetadata
from mteb.abstasks import AbsTaskRetrieval
from pathlib import Path
from typing import Any, Dict, List, Union

from evalscope.utils.logger import get_logger

logger = get_logger()

_DEFAULT_MAIN_SCORE = 'ndcg_at_10'


def build_custom_task(config: Union[Dict[str, Any], Any]):
    """Factory function to build a custom MTEB Retrieval task from user configuration.

    Args:
        config: Dict or CustomTaskConfig with keys:
            - name (str): Task name identifier (default: 'CustomRetrieval')
            - data_path (str): Path to data directory containing corpus.jsonl, queries.jsonl, qrels.jsonl
            - eval_splits (List[str]): Splits to evaluate on, default ["test"]

    Returns:
        An MTEB AbsTaskRetrieval instance ready for evaluation.
    """
    if isinstance(config, dict):
        name = config.get('name', 'CustomRetrieval')
        data_path = config.get('data_path')
        eval_splits = config.get('eval_splits', ['test'])
    else:
        name = config.name
        data_path = config.data_path
        eval_splits = config.eval_splits

    if not data_path:
        raise ValueError("'data_path' is required for custom tasks.")

    return _build_retrieval_task(name, data_path, eval_splits)


def _load_jsonl(file_path: str) -> List[Dict[str, Any]]:
    """Load data from a JSONL file.

    Args:
        file_path: Path to the JSONL file.

    Returns:
        List of dictionaries representing the records.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f'Data file not found: {file_path}')

    records = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _build_metadata(name: str, eval_splits: List[str]) -> TaskMetadata:
    """Build a TaskMetadata instance for a custom Retrieval task."""
    return TaskMetadata(
        name=name,
        description=f'Custom Retrieval task: {name}',
        reference=None,
        dataset={
            'path': name,
            'revision': 'custom'
        },
        type='Retrieval',
        category='t2t',
        modalities=['text'],
        eval_splits=eval_splits,
        eval_langs=['eng-Latn'],
        main_score=_DEFAULT_MAIN_SCORE,
        date=None,
        domains=None,
        task_subtypes=None,
        license=None,
        annotations_creators=None,
        dialect=None,
        sample_creation=None,
        bibtex_citation='',
    )


def _build_retrieval_task(
    name: str,
    data_path: str,
    eval_splits: List[str],
):
    """Build a custom Retrieval task.

    Expected files in data_path directory:
        - corpus.jsonl: {"_id": "doc1", "text": "...", "title": "..."}
        - queries.jsonl: {"_id": "q1", "text": "..."}
        - qrels.jsonl: {"query-id": "q1", "corpus-id": "doc1", "score": 1}
    """
    metadata = _build_metadata(name, eval_splits)

    class _CustomRetrieval(AbsTaskRetrieval):
        pass

    _CustomRetrieval.metadata = metadata
    _CustomRetrieval.__name__ = f'CustomRetrieval_{name}'
    _CustomRetrieval.__qualname__ = f'CustomRetrieval_{name}'

    task = _CustomRetrieval()

    # Construct file paths directly
    data_dir = Path(data_path)
    corpus_file = str(data_dir / 'corpus.jsonl')
    queries_file = str(data_dir / 'queries.jsonl')
    qrels_file = str(data_dir / 'qrels.jsonl')

    corpus_records = _load_jsonl(corpus_file)
    queries_records = _load_jsonl(queries_file)
    qrels_records = _load_jsonl(qrels_file)

    # Build corpus: {doc_id: {"text": text}}
    corpus = {}
    for idx, item in enumerate(corpus_records):
        doc_id = str(item.get('_id', '')).strip()
        if not doc_id:
            raise ValueError(f"Corpus document at index {idx} is missing '_id' field.")
        text = item.get('text', '')
        title = item.get('title', '')
        if title:
            text = f'{title} {text}'.strip()
        corpus[doc_id] = {'text': text}

    # Build queries: {qid: text}
    queries = {}
    for idx, item in enumerate(queries_records):
        qid = str(item.get('_id', '')).strip()
        if not qid:
            raise ValueError(f"Query at index {idx} is missing '_id' field.")
        queries[qid] = item.get('text', '')

    # Build relevant_docs: {qid: {doc_id: score}}
    relevant_docs = defaultdict(dict)
    for item in qrels_records:
        qid = str(item.get('query-id', ''))
        pid = str(item.get('corpus-id', ''))
        score = int(item.get('score', 1))
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
