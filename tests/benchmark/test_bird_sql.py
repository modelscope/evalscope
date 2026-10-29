import sqlite3
from pathlib import Path
from typing import Any, Dict

import pytest

import evalscope  # noqa: F401  # imported for benchmark and metric registration side effects
from evalscope.api.benchmark import DefaultDataAdapter
from evalscope.api.dataset import Sample
from evalscope.api.evaluator import TaskState
from evalscope.api.metric import SampleScore
from evalscope.api.model import ModelOutput
from evalscope.api.registry import get_benchmark
from evalscope.benchmarks.bird_sql.utils import execute_sql, extract_sql
from evalscope.config import TaskConfig

GOLD = "SELECT name FROM customers WHERE city = 'Hangzhou'"
RECORD: Dict[str, Any] = {
    'question_id': 7,
    'db_id': 'shop',
    'question': 'Which customers live in Hangzhou?',
    'evidence': 'Hangzhou refers to city = \'Hangzhou\'',
    'SQL': GOLD,
    'difficulty': 'simple',
}


@pytest.fixture()
def db_root(tmp_path: Path) -> Path:
    (tmp_path / 'shop').mkdir()
    conn = sqlite3.connect(tmp_path / 'shop' / 'shop.sqlite')
    conn.executescript(
        """
        CREATE TABLE customers(id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT, city TEXT);
        INSERT INTO customers(name, city) VALUES ('Ann','Hangzhou'),('Bob','Shanghai'),('Cat','Hangzhou');
        """
    )
    conn.commit()
    conn.close()
    return tmp_path


def _adapter(db_root: Path, **extra: Any) -> DefaultDataAdapter:
    params = {'db_root': str(db_root), **extra}
    return get_benchmark(
        'bird_sql', TaskConfig(datasets=['bird_sql'], dataset_args={'bird_sql': {'extra_params': params}})
    )


def _score(adapter: DefaultDataAdapter, sample: Sample, completion: str) -> Dict[str, Any]:
    state = TaskState(
        model='mock',
        sample=sample,
        messages=[],
        output=ModelOutput.from_content(model='mock', content=completion),
        completed=True,
    )
    return adapter.calculate_metrics(state).score


def test_registered_meta(db_root: Path) -> None:
    adapter = _adapter(db_root)

    assert adapter.subset_list == ['simple', 'moderate', 'challenging']
    assert adapter.eval_split == 'mini_dev_sqlite'
    assert adapter.metric_list == ['accuracy']
    assert adapter.few_shot_num == 0
    assert adapter.db_root == str(db_root)
    assert adapter.timeout == 30.0


def test_record_to_sample_builds_schema_prompt(db_root: Path) -> None:
    sample = _adapter(db_root).record_to_sample(dict(RECORD))

    assert sample.target == GOLD
    assert sample.subset_key == 'simple'
    assert sample.metadata['db_id'] == 'shop'
    assert 'CREATE TABLE customers' in sample.input
    assert 'sqlite_sequence' not in sample.input
    assert '-- Using valid SQLite and understanding External Knowledge' in sample.input
    assert '-- Which customers live in Hangzhou?' in sample.input
    assert "-- External Knowledge: Hangzhou refers to city = 'Hangzhou'" in sample.input
    assert sample.input.rstrip().endswith('block at the end of your answer.')


def test_sample_rows_in_schema(db_root: Path) -> None:
    sample = _adapter(db_root, sample_rows=2).record_to_sample(dict(RECORD))

    assert '2 example rows' in sample.input
    assert 'Ann' in sample.input and 'Cat' not in sample.input


@pytest.mark.parametrize(
    'completion',
    [
        f'```sql\n{GOLD};\n```',
        "Think...\n```sql\nSELECT 1\n```\nFixed:\n```sql\nSELECT c.name FROM customers c WHERE c.city LIKE 'Hang%' ORDER BY 1 DESC\n```",
        # duplicates and order are ignored, as in the official evaluation_ex.py
        "```sql\nSELECT name FROM customers WHERE city = 'Hangzhou' UNION ALL SELECT 'Ann'\n```",
    ],
)
def test_equivalent_queries_are_correct(db_root: Path, completion: str) -> None:
    adapter = _adapter(db_root)
    score = _score(adapter, adapter.record_to_sample(dict(RECORD)), completion)

    assert score.value == {'accuracy': 1.0}
    assert score.metadata['pred_error'] is None


@pytest.mark.parametrize(
    'completion',
    [
        '```sql\nSELECT name FROM customers\n```',  # superset
        "```sql\nSELECT name, city FROM customers WHERE city = 'Hangzhou'\n```",  # extra column
        "```sql\nSELECT name FROM customers WHERE city = 'Hangzhou'; DELETE FROM customers\n```",
        "```sql\nDELETE FROM customers WHERE city = 'Shanghai'\n```",
        "```sql\nATTACH DATABASE ':memory:' AS other\n```",
        'The answer is Ann and Cat.',
        '',
    ],
)
def test_wrong_or_unsafe_queries_score_zero(db_root: Path, completion: str) -> None:
    adapter = _adapter(db_root)
    score = _score(adapter, adapter.record_to_sample(dict(RECORD)), completion)

    assert score.value == {'accuracy': 0.0}
    # nothing was written to the database used by later samples
    assert execute_sql(db_root / 'shop' / 'shop.sqlite', 'SELECT COUNT(*) FROM customers').rows == [(3,)]


def test_timeout_scores_zero(db_root: Path) -> None:
    adapter = _adapter(db_root, timeout=0.5)
    endless = '```sql\nWITH RECURSIVE r(x) AS (SELECT 1 UNION ALL SELECT x + 1 FROM r) SELECT COUNT(*) FROM r\n```'
    score = _score(adapter, adapter.record_to_sample(dict(RECORD)), endless)

    assert score.value == {'accuracy': 0.0}
    assert score.metadata['pred_error'] == 'timeout'


def test_broken_reference_is_reported(db_root: Path) -> None:
    adapter = _adapter(db_root)
    sample = adapter.record_to_sample({**RECORD, 'SQL': 'SELECT nope FROM customers'})
    score = _score(adapter, sample, f'```sql\n{GOLD}\n```')

    assert score.value == {'accuracy': 0.0}
    assert score.metadata['gold_error'] is not None


def test_aggregation_by_difficulty(db_root: Path) -> None:
    adapter = _adapter(db_root)
    sample = adapter.record_to_sample(dict(RECORD))
    scores = [
        SampleScore(score=_score(adapter, sample, c), sample_id=i)
        for i, c in enumerate([f'```sql\n{GOLD}\n```', 'SELECT 1', f'```sql\n{GOLD}\n```', ''])
    ]
    aggregates = {agg.metric_name: agg for agg in adapter.aggregate_scores(scores)}

    assert aggregates['accuracy'].score == pytest.approx(0.5)
    assert aggregates['accuracy'].num == 4


def test_missing_db_root_raises(tmp_path: Path) -> None:
    adapter = _adapter(tmp_path / 'does-not-exist')
    with pytest.raises(ValueError, match='db_root does not exist'):
        adapter.load()


def test_extract_sql() -> None:
    assert extract_sql('```SQL\nSELECT 1;\n```') == 'SELECT 1'
    assert extract_sql('SELECT 2') == 'SELECT 2'
    assert extract_sql('') == ''
