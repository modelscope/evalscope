"""Temporary offline runs for the unified diagnostic pipeline."""

import json
from pathlib import Path
from typing import Callable, List, Optional

import pytest

from evalscope.api.evaluator.cache import ReviewResult
from evalscope.report import Report
from tests.analysis.helpers import review


@pytest.fixture
def make_run(tmp_path: Path) -> Callable:
    def write(name: str = 'run', rows: Optional[List[ReviewResult]] = None, repeats: int = 1) -> Path:
        root = tmp_path / name
        for folder in ['reviews/m', 'predictions/m', 'reports/m', 'configs']:
            (root / folder).mkdir(parents=True, exist_ok=True)
        records = rows if rows is not None else [review(0, 1, prediction='A'), review(1)]
        (root / 'configs/task_config.yaml').write_text(
            f'datasets: [demo]\nrepeats: {repeats}\ngeneration_config: {{max_tokens: 16}}\n'
        )
        (root / 'reviews/m/demo_default.jsonl').write_text(''.join(r.model_dump_json() + '\n' for r in records))
        (root / 'predictions/m/demo_default.jsonl').write_text(
            ''.join(
                json.dumps(
                    {
                        'index': r.index,
                        'model_output': {'choices': [{'stop_reason': 'stop'}], 'usage': {'output_tokens': 5}},
                    }
                )
                + '\n'
                for r in records
            )
        )
        (root / 'reports/m/demo.json').write_text(
            Report(
                dataset_name='demo',
                model_name='m',
                execution_summary={'requested': len(records), 'succeeded': len(records)},
            ).model_dump_json()
        )
        return root

    return write
