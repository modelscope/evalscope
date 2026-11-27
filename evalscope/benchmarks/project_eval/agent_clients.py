"""Protocol clients for L0/L1/L2 business-agent evaluation."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable, Literal, Protocol

import requests

from evalscope.project_eval.schema import EvalTask, ExecSpec


class AgentProtocolError(RuntimeError):
    """The business agent replied with a payload the contract cannot interpret."""


@dataclass
class AgentStep:
    type: str
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    result: Any = None
    elapsed_ms: float | None = None
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentTurn:
    reply: str
    steps: list[AgentStep] = field(default_factory=list)
    trace_id: str | None = None
    state: dict[str, Any] = field(default_factory=dict)
    usage: dict[str, Any] = field(default_factory=dict)
    trace_available: bool = True
    conversation: list[dict[str, str]] = field(default_factory=list)


class AgentClient(Protocol):
    def chat(self, task: EvalTask, session_id: str) -> AgentTurn: ...


Transport = Callable[..., Any]


def _messages(task: EvalTask) -> list[dict[str, str]]:
    if task.messages:
        return list(task.messages)
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
        response = requests.post(url, json=payload, timeout=timeout)
        response.raise_for_status()
        return response.json()

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
            data = self.transport(
                f'{self.base_url}/v1/eval/chat',
                {'session_id': session_id, 'message': message.get('content', '')},
                self.timeout,
            )
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
                    usage[key] += int(data['usage'].get(key, 0) or 0)
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
        response = requests.post(
            url,
            json=payload,
            headers={'Authorization': f'Bearer {self.api_key}'},
            timeout=timeout,
        )
        response.raise_for_status()
        return response.json()

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
                'input_tokens': usage.get('prompt_tokens', 0),
                'output_tokens': usage.get('completion_tokens', 0),
                'total_tokens': usage.get('total_tokens', 0),
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
        response = requests.request(method, url, json=json_body, timeout=timeout)
        response.raise_for_status()
        return response.json()

    def call(self, request: dict[str, Any]) -> AgentTurn:
        method = str(request.get('method', 'POST')).upper()
        path = str(request.get('path', ''))
        body = request.get('json') if isinstance(request.get('json'), dict) else {}
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
) -> AgentClient:
    if exec_spec.type == 'http':
        return HttpTaskClient(base_url, transport=transport)
    if exec_spec.protocol in ('rest_steps', 'otel'):
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
