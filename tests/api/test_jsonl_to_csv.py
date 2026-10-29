import csv
import json
from pathlib import Path
from typing import Any

import pytest

from evalscope.utils import jsonl_to_csv


@pytest.mark.parametrize(
    'rows, expected',
    [
        (
            [{'name': 'Alice', 'score': 1}, {'score': 2, 'name': 'Bob'}],
            [['name', 'score'], ['Alice', '1'], ['Bob', '2']],
        ),
        (
            [{'name': 'Alice', 'score': 1}, {'score': 2}],
            [['name', 'score'], ['Alice', '1'], ['', '2']],
        ),
        (
            [{'name': 'Alice'}, {'score': 2, 'name': 'Bob'}],
            [['name', 'score'], ['Alice', ''], ['Bob', '2']],
        ),
    ],
    ids=['reordered-keys', 'missing-key', 'additional-key'],
)
def test_jsonl_to_csv_aligns_fields(
    tmp_path: Path,
    rows: list[dict[str, Any]],
    expected: list[list[str]],
) -> None:
    """Keep JSON object values under the column identified by their key."""
    source = tmp_path / 'records.jsonl'
    destination = tmp_path / 'nested' / 'records.csv'
    source.write_text('\n'.join(json.dumps(row) for row in rows), encoding='utf-8')

    jsonl_to_csv(str(source), str(destination))

    with destination.open(newline='', encoding='utf-8') as stream:
        assert list(csv.reader(stream)) == expected
