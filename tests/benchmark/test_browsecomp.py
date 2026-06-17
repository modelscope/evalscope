import base64
from typing import Dict, Optional

from evalscope.api.benchmark import AgentAdapter, DataAdapter
from evalscope.api.evaluator import TaskState
from evalscope.api.model import ModelOutput
from evalscope.api.registry import get_benchmark
from evalscope.benchmarks.browsecomp.browsecomp_adapter import (
    decrypt,
    derive_key,
    normalize_answer,
    parse_judge_response,
)
from evalscope.config import TaskConfig
from evalscope.constants import JudgeStrategy


def _adapter(dataset_args: Optional[Dict] = None, limit: Optional[int] = None) -> DataAdapter:
    config = TaskConfig(
        datasets=['browsecomp'],
        dataset_args={'browsecomp': dataset_args or {}},
        limit=limit,
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
    assert isinstance(adapter, AgentAdapter)
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


def test_record_to_sample_handles_none_fields() -> None:
    adapter = _adapter()
    sample = adapter.record_to_sample({
        'problem': None,
        'answer': None,
        'problem_topic': None,
        'canary': None,
    })

    assert sample.input == ''
    assert sample.target == ''
    assert sample.metadata['problem_topic'] == ''
    assert sample.metadata['canary'] == ''


def test_load_from_modelscope_dataset_with_limit() -> None:
    adapter = _adapter(limit=1)
    dataset = adapter.load_dataset()
    sample = next(iter(dataset.values()))[0]

    assert sum(len(subset) for subset in dataset.values()) == 1
    assert sample.input[0].text
    assert sample.target
    assert sample.metadata['problem_topic']


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


def test_normalize_answer_handles_empty_or_non_string_values() -> None:
    assert normalize_answer(None) == ''
    assert normalize_answer(123) == ''
    assert normalize_answer(' Plastic-Man! ') == 'plastic man'


def test_parse_judge_response() -> None:
    assert parse_judge_response('reasoning: ok\ncorrect: yes\nconfidence: 100')
    assert parse_judge_response('reasoning: ok\ncorrect: "yes"\nconfidence: 100')
    assert not parse_judge_response('reasoning: mismatch\ncorrect: no\nconfidence: 100')
    assert not parse_judge_response('no final grade')
    assert not parse_judge_response(None)
    assert not parse_judge_response({'correct': 'yes'})
