"""Protocol clients for L0/L1/L2 business-agent evaluation."""

from __future__ import annotations

import json
from typing import Any, Callable, Literal, Protocol

import requests
from pydantic import BaseModel, ConfigDict, Field

from evalscope.project_eval.schema import EvalTask, ExecSpec, HttpRequestSpec


class AgentProtocolError(RuntimeError):
    """The business agent replied with a payload the contract cannot interpret."""


def _json_request(
    method: str,
    url: str,
    payload: dict[str, Any],
    timeout: float,
    headers: dict[str, str] | None = None,
) -> Any:
    """Send one JSON request and return the decoded JSON response."""
    response = requests.request(method, url, json=payload, headers=headers, timeout=timeout)
    response.raise_for_status()
    return response.json()


def _as_int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


class AgentStep(BaseModel):
    """One inline tool call returned by the business agent."""

    model_config = ConfigDict(extra='forbid')

    type: str
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    result: Any = None
    elapsed_ms: float | None = None
    raw: dict[str, Any] = Field(default_factory=dict)


class AgentTurn(BaseModel):
    """One aggregated business-agent turn, including multi-turn state."""

    model_config = ConfigDict(extra='forbid')

    reply: str
    steps: list[AgentStep] = Field(default_factory=list)
    trace_id: str | None = None
    state: dict[str, Any] = Field(default_factory=dict)
    usage: dict[str, Any] = Field(default_factory=dict)
    trace_available: bool = True
    conversation: list[dict[str, str]] = Field(default_factory=list)


class AgentClient(Protocol):
    """Conversation client used by `agent_api` tasks."""

    def chat(self, task: EvalTask, session_id: str) -> AgentTurn: ...


Transport = Callable[..., Any]


def _messages(task: EvalTask) -> list[dict[str, str]]:
    if task.messages:
        return [message.model_dump() for message in task.messages]
    messages: list[dict[str, str]] = []
    if task.system:
        messages.append({'role': 'system', 'content': task.system})
    messages.append({'role': 'user', 'content': task.question})
    return messages


def _steps(payload: Any) -> list[AgentStep]:
    if not isinstance(payload, list):
        raise AgentProtocolError('missing steps: expected a list')
    steps: list[AgentStep] = []
    for raw in payload:
        if not isinstance(raw, dict):
            raise AgentProtocolError('invalid step: expected an object')
        name = raw.get('name') or raw.get('tool')
        if not name:
            raise AgentProtocolError('invalid step: name is required')
        arguments = raw.get('arguments')
        if not isinstance(arguments, dict):
            raw_args = raw.get('args')
            if isinstance(raw_args, str) and raw_args.strip():
                try:
                    arguments = json.loads(raw_args)
                except ValueError:
                    arguments = {}
            else:
                arguments = {}
        steps.append(
            AgentStep(
                type=str(raw.get('type', 'tool')),
                name=str(name),
                arguments=arguments,
                result=raw.get('result'),
                elapsed_ms=raw.get('elapsed_ms') or raw.get('duration_ms'),
                raw=raw,
            )
        )
    return steps


class RestStepsClient:
    """Custom REST envelope: reply + steps/tool_calls + optional trace_id."""

    def __init__(self, base_url: str, timeout: float = 60.0, transport: Transport | None = None) -> None:
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.transport = transport or self._post

    def _post(self, url: str, payload: dict[str, Any], timeout: float) -> Any:
        return _json_request('POST', url, payload, timeout)

    def chat(self, task: EvalTask, session_id: str) -> AgentTurn:
        messages = _messages(task)
        user_messages = [message for message in messages if message.get('role') == 'user']
        if not user_messages:
            user_messages = [{'role': 'user', 'content': task.question}]

        all_steps: list[AgentStep] = []
        reply = ''
        state: dict[str, Any] = {}
        usage: dict[str, Any] = {'input_tokens': 0, 'output_tokens': 0, 'total_tokens': 0}
        trace_id: str | None = None
        conversation: list[dict[str, str]] = []
        for message in user_messages:
            payload: dict[str, Any] = {
                'session_id': session_id,
                'message': message.get('content', ''),
            }
            if task.system and not task.messages:
                payload['system'] = task.system
            data = self.transport(f'{self.base_url}/v1/eval/chat', payload, self.timeout)
            if not isinstance(data, dict) or not isinstance(data.get('reply'), str):
                raise AgentProtocolError('missing reply: expected a string')
            raw_steps = data.get('steps') or data.get('tool_calls') or data.get('trace') or []
            if raw_steps:
                all_steps.extend(_steps(raw_steps))
            reply = data['reply']
            conversation.append({'role': 'user', 'content': str(message.get('content', ''))})
            conversation.append({'role': 'assistant', 'content': reply})
            if isinstance(data.get('state'), dict):
                state.update(data['state'])
            if isinstance(data.get('usage'), dict):
                for key in usage:
                    usage[key] += _as_int(data['usage'].get(key))
            if data.get('trace_id'):
                trace_id = str(data['trace_id'])

        return AgentTurn(
            reply=reply,
            steps=all_steps,
            trace_id=trace_id,
            state=state,
            usage=usage,
            trace_available=bool(all_steps),
            conversation=conversation,
        )


class OpenAICompatClient:
    """OpenAI-compatible endpoint: final answer only, no tool-level trace."""

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str = 'EMPTY',
        timeout: float = 60.0,
        transport: Transport | None = None,
    ) -> None:
        self.base_url = base_url.rstrip('/')
        self.model = model
        self.api_key = api_key
        self.timeout = timeout
        self.transport = transport or self._post

    def _post(self, url: str, payload: dict[str, Any], timeout: float) -> Any:
        return _json_request('POST', url, payload, timeout, headers={'Authorization': f'Bearer {self.api_key}'})

    def chat(self, task: EvalTask, session_id: str) -> AgentTurn:
        data = self.transport(
            f'{self.base_url}/chat/completions',
            {'model': self.model, 'messages': _messages(task)},
            self.timeout,
        )
        try:
            reply = data['choices'][0]['message']['content']
        except (KeyError, IndexError, TypeError) as exc:
            raise AgentProtocolError('missing reply: expected choices[0].message.content') from exc
        usage = data.get('usage') if isinstance(data.get('usage'), dict) else {}
        return AgentTurn(
            reply=str(reply or ''),
            steps=[],
            trace_id=None,
            state={},
            usage={
                'input_tokens': _as_int(usage.get('prompt_tokens')),
                'output_tokens': _as_int(usage.get('completion_tokens')),
                'total_tokens': _as_int(usage.get('total_tokens')),
            },
            trace_available=False,
        )


class HttpTaskClient:
    """Direct HTTP task: call the business endpoint and treat its body as state."""

    def __init__(self, base_url: str, timeout: float = 60.0, transport: Transport | None = None) -> None:
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.transport = transport or self._request

    def _request(self, method: str, url: str, json_body: dict[str, Any], timeout: float) -> Any:
        return _json_request(method, url, json_body, timeout)

    def call(self, request: HttpRequestSpec) -> AgentTurn:
        method = request.method.upper()
        path = request.path
        body = request.json_body
        data = self.transport(method, f'{self.base_url}{path}', body, self.timeout)
        if not isinstance(data, dict):
            raise AgentProtocolError('HTTP task reply must be a JSON object')
        reply = data.get('reply') or data.get('message') or ''
        return AgentTurn(
            reply=str(reply),
            steps=[],
            trace_id=None,
            state=data,
            usage={},
            trace_available=False,
        )


def detect_capability(turn: AgentTurn) -> Literal['L2', 'L1', 'L0']:
    """Classify the delivered trace as inline (L2), id-only (L1) or answer-only (L0)."""
    if turn.steps:
        return 'L2'
    if turn.trace_id:
        return 'L1'
    return 'L0'


def build_client(
    exec_spec: ExecSpec,
    *,
    base_url: str,
    model: str = '',
    api_key: str = 'EMPTY',
    transport: Transport | None = None,
) -> AgentClient | HttpTaskClient:
    """Build the client for one task's exec envelope."""
    if exec_spec.type == 'http':
        return HttpTaskClient(base_url, transport=transport)
    if exec_spec.protocol == 'rest_steps':
        return RestStepsClient(base_url, transport=transport)
    return OpenAICompatClient(base_url, model, api_key, transport=transport)


__all__ = [
    'AgentClient',
    'AgentProtocolError',
    'AgentStep',
    'AgentTurn',
    'HttpTaskClient',
    'OpenAICompatClient',
    'RestStepsClient',
    'build_client',
    'detect_capability',
]
