"""Validated contract for externally generated business-agent eval sets."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator


class RubricCriterion(BaseModel):
    model_config = ConfigDict(extra='forbid')

    id: str
    criterion: str
    weight: float = Field(default=1.0, ge=0)
    must_pass: bool = False
    evidence_requirement: Literal['answer', 'outcome', 'process'] = 'answer'


class CheckSpec(BaseModel):
    model_config = ConfigDict(extra='forbid')

    id: str
    type: Literal['file_exists', 'file_content', 'http', 'http_json', 'sqlite', 'json_path']
    path: str = ''
    contains: str | None = None
    method: Literal['GET', 'HEAD'] = 'GET'
    url: str = ''
    expect_status: int = 200
    dsn: str = ''
    query: str = ''
    json_path: str = ''
    expect: Any = None

    @model_validator(mode='after')
    def _validate_fields(self) -> 'CheckSpec':
        if self.type in {'file_exists', 'file_content', 'json_path'} and not self.path:
            raise ValueError(f'{self.type} check requires path')
        if self.type == 'file_content' and self.contains is None:
            raise ValueError('file_content check requires contains')
        if self.type == 'json_path' and not self.json_path:
            raise ValueError('json_path check requires json_path')
        if self.type == 'http' and not self.url:
            raise ValueError('http check requires url')
        if self.type == 'http_json' and (not self.url or not self.json_path):
            raise ValueError('http_json check requires url and json_path')
        if self.type == 'http_json' and self.method != 'GET':
            raise ValueError('http_json check must use GET')
        if self.type == 'sqlite' and (not self.dsn or not self.query):
            raise ValueError('sqlite check requires dsn and query')
        return self


class TraceConfig(BaseModel):
    model_config = ConfigDict(extra='forbid')

    mode: Literal['auto', 'inline', 'fetch', 'none'] = 'auto'
    endpoint: str = ''
    header: str = 'traceparent'


class ExecSpec(BaseModel):
    model_config = ConfigDict(extra='forbid')

    type: Literal['agent_api', 'http'] = 'agent_api'
    protocol: Literal['rest_steps', 'otel', 'openai_compat'] = 'rest_steps'
    agent_url_env: str = 'EVAL_AGENT_URL'
    base_url_env: str = 'EVAL_AGENT_URL'
    reset_url: str = ''
    request: dict[str, Any] = Field(default_factory=dict)
    trace: TraceConfig = Field(default_factory=TraceConfig)
    session: dict[str, Any] = Field(default_factory=lambda: {'id_from': 'sample_id'})

    @model_validator(mode='after')
    def _validate_request(self) -> 'ExecSpec':
        if self.type == 'http' and not self.request.get('path'):
            raise ValueError('http exec requires request.path')
        return self


class Material(BaseModel):
    model_config = ConfigDict(extra='forbid')

    path: str
    sha256: str = ''


class EvalTask(BaseModel):
    model_config = ConfigDict(extra='forbid')

    id: str
    version: int = Field(default=1, ge=1)
    question: str
    system: str = ''
    messages: list[dict[str, Any]] = Field(default_factory=list)
    reference_answer: str = ''
    expected_facts: list[str] = Field(default_factory=list)
    forbidden_facts: list[str] = Field(default_factory=list)
    rubric: list[RubricCriterion] = Field(min_length=1)
    checks: list[CheckSpec] = Field(default_factory=list)
    exec: ExecSpec = Field(default_factory=ExecSpec)
    pass_threshold: float = Field(default=0.6, ge=0, le=1)
    source_refs: list[str] = Field(default_factory=list)
    origin: Literal['fact', 'inference'] = 'fact'
    materials: list[Material] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode='after')
    def _validate_task(self) -> 'EvalTask':
        if not self.question and not self.messages:
            raise ValueError('task requires question or messages')
        return self


def load_eval_set(path: str | Path) -> list[EvalTask]:
    """Load and validate a JSONL eval set, rejecting duplicate ids."""
    source = Path(path)
    tasks: list[EvalTask] = []
    seen: set[str] = set()
    for line_number, line in enumerate(source.read_text(encoding='utf-8').splitlines(), 1):
        if not line.strip():
            continue
        try:
            data = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f'{source}:{line_number}: invalid JSON: {exc}') from exc
        task = EvalTask.model_validate(data)
        if task.id in seen:
            raise ValueError(f'duplicate task id: {task.id}')
        seen.add(task.id)
        tasks.append(task)
    if not tasks:
        raise ValueError(f'eval set is empty: {source}')
    return tasks


__all__ = [
    'CheckSpec',
    'EvalTask',
    'ExecSpec',
    'Material',
    'RubricCriterion',
    'TraceConfig',
    'load_eval_set',
]
