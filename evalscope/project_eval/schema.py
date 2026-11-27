"""Validated contract for externally generated business-agent eval sets."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

RUNTIME_CHECKS_KEY = 'runtime_checks'
AGENT_CAPABILITY_KEY = 'agent_capability'
AGENT_STATE_KEY = 'agent_state'
PROJECT_ROOT_KEY = 'project_root'


class RubricCriterion(BaseModel):
    """One rubric criterion; the Judge returns one verdict for each."""

    model_config = ConfigDict(extra='forbid')

    id: str
    criterion: str
    weight: float = Field(default=1.0, ge=0)
    must_pass: bool = False
    evidence_requirement: Literal['answer', 'outcome', 'process'] = 'answer'


class CheckSpec(BaseModel):
    """One read-only end-state check and the value it expects."""

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
        if self.type in {'http', 'http_json'}:
            parsed = urlsplit(self.url)
            if parsed.scheme not in {'http', 'https'} or not parsed.hostname:
                raise ValueError(f'{self.type} check requires an http(s) url')
            try:
                parsed.port
            except ValueError as exc:
                raise ValueError(f'{self.type} check has an invalid port') from exc
        if self.type == 'http_json' and self.method != 'GET':
            raise ValueError('http_json check must use GET')
        if self.type == 'sqlite' and (not self.dsn or not self.query):
            raise ValueError('sqlite check requires dsn and query')
        return self


class HttpRequestSpec(BaseModel):
    """Request envelope for `exec.type='http'` tasks."""

    model_config = ConfigDict(extra='forbid', populate_by_name=True)

    method: str = 'POST'
    path: str = ''
    json_body: dict[str, Any] = Field(default_factory=dict, alias='json')


class SessionConfig(BaseModel):
    """Business session key contract.

    `sample_id` is the JSONL task `id`, which is also the key the business agent
    uses for its audit/state.
    """

    model_config = ConfigDict(extra='forbid')

    id_from: Literal['sample_id'] = 'sample_id'


class MessageSpec(BaseModel):
    """One user turn of an optional multi-turn script."""

    model_config = ConfigDict(extra='forbid')

    role: Literal['user'] = 'user'
    content: str


class TraceConfig(BaseModel):
    """Trace delivery contract.

    `auto` uses inline steps when the Agent returns them, `inline` requires them,
    and `none` disables trace evidence.
    """

    model_config = ConfigDict(extra='forbid')

    mode: Literal['auto', 'inline', 'none'] = 'auto'


class ExecSpec(BaseModel):
    """How EvalScope calls the business Agent for one task."""

    model_config = ConfigDict(extra='forbid')

    type: Literal['agent_api', 'http'] = 'agent_api'
    protocol: Literal['rest_steps', 'openai_compat'] = 'rest_steps'
    agent_url_env: str = 'EVAL_AGENT_URL'
    base_url_env: str = 'EVAL_AGENT_URL'
    reset_url: str = ''
    request: HttpRequestSpec = Field(default_factory=HttpRequestSpec)
    trace: TraceConfig = Field(default_factory=TraceConfig)
    session: SessionConfig = Field(default_factory=SessionConfig)

    @model_validator(mode='after')
    def _validate_request(self) -> 'ExecSpec':
        if self.type == 'http' and not self.request.path:
            raise ValueError('http exec requires request.path')
        return self


class ProjectEvalRunConfig(BaseModel):
    """Typed dataset_args.project_eval configuration for the adapter."""

    model_config = ConfigDict(extra='forbid')

    local_path: str = ''
    agent_url: str = ''
    project_root: str = ''
    check_url_prefixes: list[str] = Field(default_factory=list)

    @field_validator('check_url_prefixes')
    @classmethod
    def _validate_prefixes(cls, prefixes: list[str]) -> list[str]:
        for prefix in prefixes:
            parsed = urlsplit(prefix)
            if parsed.scheme not in {'http', 'https'} or not parsed.hostname:
                raise ValueError(f'invalid check_url_prefix: {prefix!r}')
            try:
                parsed.port
            except ValueError as exc:
                raise ValueError(f'invalid check_url_prefix port: {prefix!r}') from exc
            if parsed.username or parsed.password:
                raise ValueError('check_url_prefix must not contain user info')
            if parsed.query or parsed.fragment:
                raise ValueError('check_url_prefix must not contain a query or fragment')
        return prefixes


class Material(BaseModel):
    """One source material supplied by the external generator."""

    model_config = ConfigDict(extra='forbid')

    path: str = Field(min_length=1)
    sha256: str = Field(default='', pattern=r'^(?:[0-9a-f]{64})?$')


class EvalTask(BaseModel):
    """One externally generated business-agent eval task.

    `question` is required; `messages` are an optional multi-turn supplement.
    `version`, `source_refs`, `origin` and `materials` are provenance fields
    validated for the external generator; execution does not read them.
    """

    model_config = ConfigDict(extra='forbid')

    id: str = Field(min_length=1)
    version: int = Field(default=1, ge=1)
    question: str = Field(min_length=1)
    system: str = ''
    messages: list[MessageSpec] = Field(default_factory=list)
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
    'AGENT_CAPABILITY_KEY',
    'AGENT_STATE_KEY',
    'CheckSpec',
    'EvalTask',
    'ExecSpec',
    'HttpRequestSpec',
    'Material',
    'MessageSpec',
    'PROJECT_ROOT_KEY',
    'ProjectEvalRunConfig',
    'RUNTIME_CHECKS_KEY',
    'RubricCriterion',
    'SessionConfig',
    'TraceConfig',
    'load_eval_set',
]
