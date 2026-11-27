import json
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from evalscope.project_eval.schema import ProjectEvalRunConfig, load_eval_set


def write_jsonl(tmp_path: Path, rows: list[dict[str, Any]]) -> Path:
    path = tmp_path / 'eval_set.v1.jsonl'
    path.write_text('\n'.join(json.dumps(r, ensure_ascii=False) for r in rows), encoding='utf-8')
    return path


def task(**overrides: Any) -> dict[str, Any]:
    row = {
        'id': 't1',
        'version': 1,
        'question': '查余额',
        'system': '客服',
        'messages': [],
        'rubric': [{
            'id': 'c1',
            'criterion': '回答正确',
            'must_pass': True,
            'evidence_requirement': 'outcome',
        }],
        'checks': [{
            'id': 'j-1',
            'type': 'json_path',
            'path': 'state.json',
            'json_path': 'balance_cent',
            'expect': 200000,
        }],
        'exec': {
            'type': 'agent_api',
            'protocol': 'rest_steps',
            'agent_url_env': 'EVAL_AGENT_URL',
            'trace': {'mode': 'auto'},
            'session': {'id_from': 'sample_id'},
        },
        'source_refs': ['internal/agent/chat.go:123'],
        'origin': 'fact',
        'materials': [],
        'metadata': {},
    }
    row.update(overrides)
    return row


def test_loads_valid_jsonl(tmp_path: Path) -> None:
    tasks = load_eval_set(write_jsonl(tmp_path, [task()]))
    assert tasks[0].id == 't1'
    assert tasks[0].checks[0].type == 'json_path'


def test_loads_multi_turn_messages(tmp_path: Path) -> None:
    messages = [{'role': 'user', 'content': '查余额'}, {'role': 'user', 'content': '谢谢'}]
    tasks = load_eval_set(write_jsonl(tmp_path, [task(messages=messages)]))
    assert len(tasks[0].messages) == 2


def test_rejects_duplicate_ids(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match='duplicate task id'):
        load_eval_set(write_jsonl(tmp_path, [task(), task()]))


def test_rejects_empty_question(tmp_path: Path) -> None:
    with pytest.raises(ValidationError):
        load_eval_set(write_jsonl(tmp_path, [task(question='')]))


def test_rejects_empty_id(tmp_path: Path) -> None:
    with pytest.raises(ValidationError):
        load_eval_set(write_jsonl(tmp_path, [task(id='')]))


def test_rejects_unknown_evidence_requirement(tmp_path: Path) -> None:
    with pytest.raises(ValidationError):
        load_eval_set(write_jsonl(tmp_path, [task(rubric=[{
            'id': 'c1',
            'criterion': 'ok',
            'evidence_requirement': 'vibes',
        }])]))


def test_rejects_empty_rubric(tmp_path: Path) -> None:
    with pytest.raises(ValidationError):
        load_eval_set(write_jsonl(tmp_path, [task(rubric=[])]))


def test_rejects_incomplete_check_specs(tmp_path: Path) -> None:
    with pytest.raises(ValidationError):
        load_eval_set(write_jsonl(tmp_path, [task(checks=[{'id': 'h', 'type': 'http'}])]))
    with pytest.raises(ValidationError):
        load_eval_set(write_jsonl(tmp_path, [task(checks=[{'id': 's', 'type': 'sqlite'}])]))
    with pytest.raises(ValidationError):
        load_eval_set(write_jsonl(tmp_path, [task(checks=[{'id': 'j', 'type': 'json_path'}])]))


def test_rejects_http_exec_without_request_path(tmp_path: Path) -> None:
    with pytest.raises(ValidationError):
        load_eval_set(write_jsonl(tmp_path, [task(exec={'type': 'http', 'request': {}})]))


def test_rejects_out_of_range_threshold(tmp_path: Path) -> None:
    with pytest.raises(ValidationError):
        load_eval_set(write_jsonl(tmp_path, [task(pass_threshold=1.5)]))


def test_reference_fields_default_and_round_trip(tmp_path: Path) -> None:
    loaded = load_eval_set(write_jsonl(tmp_path, [task()]))[0]
    assert loaded.reference_answer == ''
    assert loaded.expected_facts == []
    assert loaded.forbidden_facts == []

    enriched = task(reference_answer='余额 2000 元', expected_facts=['2000 元'],
                    forbidden_facts=['9999 元'])
    loaded = load_eval_set(write_jsonl(tmp_path, [enriched]))[0]
    assert loaded.reference_answer == '余额 2000 元'
    assert loaded.expected_facts == ['2000 元']


def test_run_config_rejects_unsafe_http_prefixes() -> None:
    for prefix in ['', 'http://user@state.example', 'http://state.example?x=1']:
        with pytest.raises(ValidationError):
            ProjectEvalRunConfig(check_url_prefixes=[prefix])
    config = ProjectEvalRunConfig(check_url_prefixes=['https://state.example'])
    assert config.check_url_prefixes == ['https://state.example']
