import base64
import csv
from pathlib import Path
from typing import Dict, Optional

from evalscope.api.benchmark import DataAdapter
from evalscope.api.evaluator import TaskState
from evalscope.api.model import ModelOutput
from evalscope.api.registry import get_benchmark
from evalscope.benchmarks.browsecomp.browsecomp_adapter import decrypt, derive_key, parse_judge_response
from evalscope.config import TaskConfig
from evalscope.constants import JudgeStrategy


def _adapter(dataset_args: Optional[Dict] = None) -> DataAdapter:
    config = TaskConfig(
        datasets=['browsecomp'],
        dataset_args={'browsecomp': dataset_args or {}},
        judge_strategy=JudgeStrategy.RULE,
    )
    return get_benchmark('browsecomp', config=config)


def _encrypt(plaintext: str, password: str) -> str:
    encoded = plaintext.encode()
    key = derive_key(password, len(encoded))
    encrypted = bytes(a ^ b for a, b in zip(encoded, key))
    return base64.b64encode(encrypted).decode()


def test_decrypt_round_trip() -> None:
    canary = 'browsecomp:test-canary'
    assert decrypt(_encrypt('Plastic Man', canary), canary) == 'Plastic Man'


def test_record_to_sample_decrypts_official_fields() -> None:
    adapter = _adapter()
    canary = 'browsecomp:test-canary'
    sample = adapter.record_to_sample({
        'problem': _encrypt('Who occasionally breaks the fourth wall?', canary),
        'answer': _encrypt('Plastic Man', canary),
        'problem_topic': 'Art',
        'canary': canary,
    })

    assert sample.input == 'Who occasionally breaks the fourth wall?'
    assert sample.target == 'Plastic Man'
    assert sample.metadata['problem_topic'] == 'Art'


def test_load_from_local_csv(tmp_path: Path) -> None:
    canary = 'browsecomp:test-canary'
    data_path = tmp_path / 'browse_comp_test_set.csv'
    with data_path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['problem', 'answer', 'problem_topic', 'canary'])
        writer.writeheader()
        writer.writerow({
            'problem': _encrypt('Question one?', canary),
            'answer': _encrypt('Answer one', canary),
            'problem_topic': 'Science',
            'canary': canary,
        })

    adapter = _adapter({'local_path': str(data_path)})
    dataset = adapter.load_dataset()
    sample = dataset['default'][0]

    assert len(dataset['default']) == 1
    assert sample.input[0].text.startswith('Question one?')
    assert sample.target == 'Answer one'


def test_extract_answer_and_rule_score() -> None:
    adapter = _adapter()
    sample = adapter.record_to_sample({
        'problem': _encrypt('Question?', 'canary'),
        'answer': _encrypt('Plastic Man', 'canary'),
        'problem_topic': 'Art',
        'canary': 'canary',
    })
    task_state = TaskState(
        model='mock',
        sample=sample,
        output=ModelOutput.from_content(
            model='mock',
            content='Explanation: searched evidence\nExact Answer: plastic man\nConfidence: 90%',
        ),
        completed=True,
    )

    result = adapter.calculate_metrics(task_state)

    assert result.score.value['is_correct'] == 1.0
    assert result.score.value['is_incorrect'] == 0.0
    assert result.score.extracted_prediction == 'plastic man'
    assert result.score.main_score_name == 'is_correct'


def test_parse_judge_response() -> None:
    assert parse_judge_response('reasoning: ok\ncorrect: yes\nconfidence: 100')
    assert parse_judge_response('reasoning: ok\ncorrect: "yes"\nconfidence: 100')
    assert not parse_judge_response('reasoning: mismatch\ncorrect: no\nconfidence: 100')
    assert not parse_judge_response('no final grade')
