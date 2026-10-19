# BIRD Mini-Dev adapter: text-to-SQL on real SQLite databases, scored by execution accuracy.
#
# Paper: https://arxiv.org/abs/2305.03111 (NeurIPS 2023 Datasets and Benchmarks)
# Dataset: https://huggingface.co/datasets/birdsql/bird_mini_dev (CC BY-SA 4.0)
# Databases: https://github.com/bird-bench/mini_dev (minidev.zip)

import zipfile
from pathlib import Path
from typing import Any, Dict, Optional

from evalscope.api.benchmark import BenchmarkMeta, DefaultDataAdapter
from evalscope.api.dataset import Sample
from evalscope.api.evaluator import TaskState
from evalscope.api.metric import Score
from evalscope.api.registry import register_benchmark
from evalscope.constants import DEFAULT_EVALSCOPE_CACHE_DIR, Tags
from evalscope.utils.download_utils import download_url
from evalscope.utils.logger import get_logger

from .utils import build_prompt, db_file, execute_sql, extract_sql, schema_prompt

logger = get_logger()

SUBSET_LIST = ['simple', 'moderate', 'challenging']
DB_ARCHIVE_URL = 'https://bird-bench.oss-cn-beijing.aliyuncs.com/minidev.zip'
DB_ARCHIVE_SHA256 = 'cc48ba16838204e4e214512030cb572eeb5f7bcdd999bae4b9b6ff12ec13b92f'  # official minidev.zip
DEFAULT_CACHE_SUBDIR = 'bird_sql'

DESCRIPTION = """
## Overview

BIRD (BIg Bench for LaRge-scale Database Grounded Text-to-SQL) evaluates whether a model can turn a
natural-language question into an SQL query over real, large and "dirty" databases. This benchmark
uses BIRD Mini-Dev, 500 high-quality SQLite instances selected from the BIRD dev set over 11
databases.

## Task Description

- **Task Type**: Text-to-SQL
- **Input**: The `CREATE TABLE` statements of the database, the question, and BIRD's external
  knowledge (evidence) for the question
- **Output**: An SQLite query in a ```sql code block
- **Domain**: Databases across 11 domains (finance, sports, healthcare, education, ...)

## Key Features

- 500 questions in three difficulty levels: simple 148, moderate 250, challenging 102
- Real databases with noisy values; the external knowledge explains business definitions
- Scored by executing the predicted and reference queries, not by string match

## Evaluation Notes

- `accuracy` (primary) is BIRD execution accuracy (EX): the prediction is correct when the set of
  rows it returns equals the set returned by the reference SQL, as in the official
  `evaluation_ex.py` (row order and duplicates are ignored; column order matters)
- Queries run read-only with one statement per query and a 30 s timeout (`timeout`); a query that
  writes, attaches a database, or exceeds the timeout scores 0
- The prompt follows the official baseline (schema + question + external knowledge + "think step by
  step"), but asks for the final query in a ```sql block so reasoning models can think first;
  `sample_rows` adds example rows per table as in the official `num_rows` option
- The databases (`minidev.zip`, about 800 MB) are downloaded and cached on first use; set
  `db_root` to an existing `dev_databases` directory to skip the download
- Subsets are the difficulty levels; `limit` applies per subset
- Sample metadata records `pred_error` (why the prediction failed to run) and `gold_empty` (the
  reference returns no rows, so any query returning nothing also matches)
- [Paper](https://arxiv.org/abs/2305.03111) | [Code](https://github.com/bird-bench/mini_dev) |
  [Dataset](https://huggingface.co/datasets/birdsql/bird_mini_dev) | [Website](https://bird-bench.github.io/)
"""


@register_benchmark(
    BenchmarkMeta(
        name='bird_sql',
        pretty_name='BIRD-SQL Mini-Dev',
        dataset_id='evalscope/bird_mini_dev',
        hf_dataset_id='birdsql/bird_mini_dev',
        paper_url='https://arxiv.org/abs/2305.03111',
        tags=[Tags.CODING, Tags.REASONING],
        description=DESCRIPTION,
        subset_list=SUBSET_LIST,
        eval_split='mini_dev_sqlite',
        few_shot_mode='disabled',
        metric_list=['accuracy'],
        prompt_template='{question}',
        evaluation_version='v1.0',
        extra_params={
            'db_root': {
                'type': 'str | null',
                'description': 'Directory with one <db_id>/<db_id>.sqlite per database (the `dev_databases` '
                'folder of minidev.zip); if null, minidev.zip is downloaded and cached.',
                'value': None,
            },
            'timeout': {
                'type': 'float',
                'description': 'Per-query execution timeout in seconds (official evaluator: 30).',
                'value': 30.0,
            },
            'sample_rows': {
                'type': 'int',
                'description': 'Number of example rows per table to add to the schema prompt (official default: 0).',
                'value': 0,
            },
        },
    )
)
class BirdSQLAdapter(DefaultDataAdapter):
    """Data adapter for BIRD Mini-Dev (SQLite), scored by execution accuracy."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        # Load the single split once and split it into one subset per difficulty level.
        self.reformat_subset = True
        self.db_root: Optional[str] = self.extra_params.get('db_root')
        self.timeout = float(self.extra_params.get('timeout', 30.0))
        self.sample_rows = int(self.extra_params.get('sample_rows', 0))
        self._schema_cache: Dict[str, str] = {}

    def load(self):
        if not self.db_root:
            self.db_root = str(self._ensure_databases())
        elif not Path(self.db_root).is_dir():
            raise ValueError(f'BIRD db_root does not exist: {self.db_root}')
        return super().load()

    def _ensure_databases(self) -> Path:
        """Download and extract minidev.zip once; return its ``dev_databases`` directory."""
        cache_root = Path(DEFAULT_EVALSCOPE_CACHE_DIR) / DEFAULT_CACHE_SUBDIR
        found = _find_db_root(cache_root)
        if found:
            logger.info(f'BIRD databases found: {found}')
            return found
        zip_path = cache_root / 'minidev.zip'
        try:
            # Atomic download with retries, progress bar and checksum verification.
            download_url(DB_ARCHIVE_URL, str(zip_path), sha256=DB_ARCHIVE_SHA256)
            with zipfile.ZipFile(zip_path) as zf:
                # Only the SQLite dev_databases are needed; skip the MySQL/PostgreSQL variants (~2 GB).
                zf.extractall(cache_root, members=[m for m in zf.namelist() if '/MINIDEV/dev_databases/' in m])
        except Exception as e:
            raise ValueError(
                f'Failed to download or extract BIRD databases: {e}. '
                f'Download {DB_ARCHIVE_URL} manually and set extra_params["db_root"] to its dev_databases folder.'
            ) from e
        finally:
            if zip_path.exists():
                zip_path.unlink()
        found = _find_db_root(cache_root)
        if not found:
            raise ValueError(f'Extracted {DB_ARCHIVE_URL} but found no dev_databases folder under {cache_root}')
        return found

    def _schema(self, db_id: str) -> str:
        if db_id not in self._schema_cache:
            self._schema_cache[db_id] = schema_prompt(db_file(self.db_root, db_id), self.sample_rows)
        return self._schema_cache[db_id]

    def record_to_sample(self, record: Dict[str, Any]) -> Sample:
        db_id = record['db_id']
        evidence = record.get('evidence') or None
        return Sample(
            input=build_prompt(self._schema(db_id), record['question'], evidence),
            target=record['SQL'],
            subset_key=record['difficulty'],
            metadata={
                'question_id': record['question_id'],
                'db_id': db_id,
                'question': record['question'],
                'evidence': evidence,
                'difficulty': record['difficulty'],
            },
        )

    def extract_answer(self, prediction: str, task_state: TaskState) -> str:
        return extract_sql(prediction)

    def match_score(
        self, original_prediction: str, filtered_prediction: str, reference: str, task_state: TaskState
    ) -> Score:
        path = db_file(self.db_root, task_state.metadata['db_id'])
        gold = execute_sql(path, reference, self.timeout)
        pred = execute_sql(path, filtered_prediction, self.timeout)
        if not gold.ok:
            logger.warning(f'BIRD reference SQL failed (question {task_state.metadata["question_id"]}): {gold.error}')
        correct = gold.ok and pred.ok and set(pred.rows) == set(gold.rows)
        return Score(
            extracted_prediction=filtered_prediction,
            prediction=original_prediction,
            value={'accuracy': float(correct)},
            main_score_name='accuracy',
            metadata={
                'pred_error': pred.error,
                'gold_error': gold.error,
                'gold_empty': gold.ok and not gold.rows,
            },
        )


def _find_db_root(root: Path) -> Optional[Path]:
    """Return the first ``dev_databases`` directory under ``root`` that contains SQLite databases."""
    if not root.exists():
        return None
    for candidate in sorted(root.rglob('dev_databases')):
        if candidate.is_dir() and any(candidate.glob('*/*.sqlite')):
            return candidate
    return None
