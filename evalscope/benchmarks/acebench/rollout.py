# Copyright (c) Alibaba, Inc. and its affiliates.
"""Agent rollouts for ACEBench.

Ports the official runners (``model_inference/apimodel_inference.py``, ``multi_step`` and
``multi_turn``): the agent talks to a simulated environment, every message it addresses to the
executor is recorded as the process trace, and the resulting environment state is what gets graded.

Two deliberate deviations from upstream, neither of which changes a score:

* Calls are dispatched with ``getattr(instance, name)(**kwargs)`` instead of ``eval()`` on a
  reconstructed call string, so model output is never executed as code.
* API instances are kept per rollout instead of in module globals, which makes concurrent
  evaluation safe.

The loop also records an :class:`AgentTrace` and keeps the model's own assistant messages, so the
web dashboard renders these samples the same way it renders AgentLoop-driven benchmarks and
``PerfCollector`` picks up per-turn latency and token counts.
"""

import json
import re
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from evalscope.api.agent import AgentTrace, EventType
from evalscope.api.messages import ChatMessage, ChatMessageSystem, ChatMessageTool, ChatMessageUser
from evalscope.api.model import Model, ModelOutput, ModelUsage
from evalscope.utils.logger import get_logger

from .parser import decode_execution_calls
from .prompts import agent_prompt_set, build_agent_prompts, build_user_simulator_prompt
from .scenarios import load_scenario_instances, snapshot_states

logger = get_logger()

FINISH_MARKER = 'finish conversation'

#: ``AgentTrace.framework`` value for this benchmark's own loop. Not ``'native'``: these rollouts
#: are driven here rather than by :class:`evalscope.api.agent.AgentLoop`.
TRACE_FRAMEWORK = 'acebench'

#: Sentence upstream uses to nudge a multi-step agent whose message did not decode into calls.
DECODE_ERROR_NUDGE = 'Please do not ask me any questions, use the known conditions to solve the problem'


@dataclass
class RolloutResult:
    """Outcome of an agent rollout."""

    process: List[str] = field(default_factory=list)
    """Messages the agent sent to the executor, in order, as ACEBench records them."""

    end_state: List[Dict[str, Any]] = field(default_factory=list)
    """Graded attributes of every involved API class after the rollout."""

    messages: List[ChatMessage] = field(default_factory=list)
    """Transcript of the rollout, for inspection in the review files.

    Assistant entries are the model's own :class:`ChatMessage` objects rather than copies, so the
    ``perf_metrics`` the model API attached to them survive into ``PerfCollector``.
    """

    usage: ModelUsage = field(default_factory=ModelUsage)
    """Token usage accumulated over the rollout."""

    trace: AgentTrace = field(default_factory=AgentTrace)
    """Structured trajectory consumed by the dashboard's trace view."""


@dataclass
class _Execution:
    """One executor turn: what the agent asked for and what came back."""

    observation: Any
    """Value handed back to the agent, with upstream's exact shape."""

    calls: List[Dict[str, Dict[str, Any]]] = field(default_factory=list)
    """Decoded calls, empty when the message did not decode."""

    outcomes: List[Any] = field(default_factory=list)
    """One outcome per entry in :attr:`calls`."""

    decode_error: Optional[str] = None
    """Why decoding failed, when it did."""


def run_rollout(
    model: Model,
    metadata: Dict[str, Any],
    max_steps: int = 40,
    user_model: Optional[Model] = None,
) -> RolloutResult:
    """Run the rollout that matches the sample's agent category.

    Args:
        model: Model under evaluation, playing the agent.
        metadata: Sample metadata with ``test_category``, ``question``, ``functions``,
            ``initial_config``, ``involved_classes`` and ``language``.
        max_steps: Maximum number of loop iterations.
        user_model: Model playing the user; required by ``agent_multi_turn``.

    Returns:
        The recorded process trace, end state and trajectory.
    """
    test_category = metadata.get('test_category', '')
    instances = load_scenario_instances(
        initial_config=metadata.get('initial_config') or {},
        involved_classes=metadata.get('involved_classes') or [],
        language=metadata.get('language', 'en'),
    )
    result = RolloutResult(
        trace=AgentTrace(framework=TRACE_FRAMEWORK, strategy=test_category or None, max_steps=max_steps)
    )

    if 'multi_step' in test_category:
        _run_multi_step(model, metadata, instances, max_steps, result)
    else:
        if user_model is None:
            raise ValueError(
                'agent_multi_turn needs a user simulator; set extra_params.user_model to the model '
                'that should play the user (the official runner uses gpt-4o).'
            )
        _run_multi_turn(model, user_model, metadata, instances, max_steps, result)

    result.end_state = snapshot_states(instances)
    return result


def _run_multi_step(
    model: Model,
    metadata: Dict[str, Any],
    instances: Dict[str, Any],
    max_steps: int,
    result: RolloutResult,
) -> None:
    """Drive the agent-executor loop of ``agent_multi_step``."""
    transcript = ''
    # ACEBench seeds the history with the task description as a user message.
    question = metadata.get('question', '')
    history: List[Tuple[str, Any]] = [('user', question)]
    result.messages.append(ChatMessageUser(content=f'user: {question}'))

    # Trace step, which advances once per agent turn so that a turn and the execution of its calls
    # land in the same group. The loop index cannot be used: upstream spends a separate iteration on
    # the executor, which would split every turn across two groups in the dashboard.
    turn = 0

    for step in range(max_steps):
        last_sender = history[-1][0]
        if step == 0 or last_sender == 'execution':
            # Upstream calls ``Mulit_Step_Scene.get_inference_message`` only in this branch (see
            # ``apimodel_inference.multi_step_inference``), so the transcript only ever grows by a
            # user or execution line and the agent never sees its own previous messages. Its
            # ``sender == 'agent'`` case is dead code here and only reachable from multi_turn.
            # Reproduced deliberately: a model that needs that history will loop on the same call
            # until the step budget runs out, and that is what the official numbers measure.
            transcript += _transcript_line(history[-1], execution_label='execution result')
            message = _agent_step(model, metadata, transcript, result, turn)
            history.append(('agent', message))
        else:
            result.process.append(history[-1][1])
            execution = _execute(history[-1][1], instances, catch_decode_error=True)
            _record_execution(result, turn, execution)
            history.append(('execution', execution.observation))
            turn += 1

        if step > 1 and _is_finished(history[-1][1]):
            _record_finish(result, turn)
            return

    _record_max_steps(result, turn)


def _run_multi_turn(
    model: Model,
    user_model: Model,
    metadata: Dict[str, Any],
    instances: Dict[str, Any],
    max_steps: int,
    result: RolloutResult,
) -> None:
    """Drive the user-agent-executor loop of ``agent_multi_turn``."""
    language = metadata.get('language', 'en')
    involved_classes = metadata.get('involved_classes') or []
    templates = agent_prompt_set(language)

    simulator = _UserSimulator(
        model=user_model,
        system_prompt=build_user_simulator_prompt(metadata.get('question', ''), involved_classes, language),
        opening=templates['user_opening'],
    )

    transcript = ''
    # The opening user message precedes any agent turn, so it carries no trace event and the
    # dashboard shows it as the conversation's preamble.
    history: List[Tuple[str, Any]] = [('user', simulator.start(result, step=None))]
    # Recipient of the last message, which is what upstream dispatches on.
    recipient = 'agent'
    turn = 0

    for step in range(max_steps):
        transcript += _transcript_line(history[-1], execution_label='execution')
        if recipient == 'user':
            message = simulator.respond(history[-1][1], result, step=turn)
            history.append(('user', message))
            recipient = 'agent'
            turn += 1
        elif recipient == 'agent':
            message = _agent_step(model, metadata, transcript, result, turn)
            history.append(('agent', message))
            recipient = 'execution' if _looks_like_calls(message) else 'user'
        else:
            result.process.append(history[-1][1])
            execution = _execute(history[-1][1], instances, catch_decode_error=False)
            _record_execution(result, turn, execution)
            history.append(('execution', execution.observation))
            recipient = 'agent'
            turn += 1

        if step > 1 and _is_finished(history[-1][1]):
            _record_finish(result, turn)
            return

    _record_max_steps(result, turn)


class _UserSimulator:
    """Model that plays the user, mirroring ``APIModel_user.APIUSER``."""

    def __init__(self, model: Model, system_prompt: str, opening: str) -> None:
        self.model = model
        # Upstream records its own turns with the ``system`` role; kept so the simulator sees the
        # same conversation shape it does officially.
        self.messages: List[ChatMessage] = [
            ChatMessageSystem(content=system_prompt),
            ChatMessageUser(content=opening),
        ]

    def start(self, result: RolloutResult, step: Optional[int]) -> str:
        """Produce the opening user message."""
        return self._generate(result, step)

    def respond(self, agent_message: Any, result: RolloutResult, step: Optional[int]) -> str:
        """Answer the agent's latest message."""
        self.messages.append(ChatMessageUser(content=_strip_role_prefix(str(agent_message))))
        return self._generate(result, step)

    def _generate(self, result: RolloutResult, step: Optional[int]) -> str:
        output = self.model.generate(self.messages)
        if output.usage is not None:
            result.usage += output.usage
        response = output.completion or ''
        self.messages.append(ChatMessageSystem(content=response))

        # The simulated user is recorded as a user message so that ``PerfCollector``, which only
        # reads assistant messages, keeps reporting the evaluated model's latency rather than this
        # second model's.
        message = ChatMessageUser(content=f'user: {response}')
        result.messages.append(message)
        if step is not None:
            # A reply shares its step with the agent turn it answers, so that step carries two
            # ``model_generate`` events. The dashboard takes the first one as the step's assistant
            # header, which is correct only because the agent turn is always recorded first --
            # keep that order if this loop is ever restructured.
            result.trace.add_event(
                step=step,
                type=EventType.MODEL_GENERATE,
                message_id=message.id,
                token_usage=_usage_dict(output.usage),
                payload={'source': 'user_simulator', 'model': self.model.name},
            )
        return response


def _agent_step(
    model: Model,
    metadata: Dict[str, Any],
    transcript: str,
    result: RolloutResult,
    step: int,
) -> str:
    """Ask the model under evaluation for its next message."""
    system_prompt, user_prompt = build_agent_prompts(
        transcript=transcript,
        functions=metadata.get('functions') or [],
        involved_classes=metadata.get('involved_classes') or [],
        test_category=metadata.get('test_category', ''),
        language=metadata.get('language', 'en'),
    )
    started = time.monotonic()
    output = model.generate([ChatMessageSystem(content=system_prompt), ChatMessageUser(content=user_prompt)])
    latency_ms = (time.monotonic() - started) * 1000
    if output.usage is not None:
        result.usage += output.usage

    # Keep the model's own message: it carries the ``perf_metrics`` the model API attached.
    message = output.message
    result.messages.append(message)
    if result.trace.total_usage is None:
        result.trace.total_usage = output.usage
    elif output.usage is not None:
        result.trace.total_usage += output.usage
    result.trace.add_event(
        step=step,
        type=EventType.MODEL_GENERATE,
        message_id=message.id,
        latency_ms=latency_ms,
        token_usage=_usage_dict(output.usage),
        payload={'stop_reason': output.stop_reason},
    )
    return output.completion or ''


def _record_execution(result: RolloutResult, step: int, execution: _Execution) -> None:
    """Append the executor's observations and trace the calls that produced them.

    Executing the simulated APIs is an in-process dict update, so no per-call latency is recorded;
    the step's wall-clock time is dominated by the preceding model turn.
    """
    if execution.decode_error is not None:
        message = ChatMessageUser(content=f'execution: {execution.observation}')
        result.messages.append(message)
        result.trace.add_event(
            step=step,
            type=EventType.ERROR,
            message_id=message.id,
            payload={'source': 'parse', 'message': execution.decode_error},
        )
        return

    for call, outcome in zip(execution.calls, execution.outcomes):
        name, arguments = next(iter(call.items()))
        call_id = uuid.uuid4().hex[:8]
        result.trace.add_event(
            step=step,
            type=EventType.TOOL_CALL,
            payload={'id': call_id, 'name': name, 'arguments': arguments},
        )
        observation = _serialize(outcome)
        message = ChatMessageTool(content=observation, tool_call_id=call_id, function=name)
        result.messages.append(message)
        result.trace.add_event(
            step=step,
            type=EventType.TOOL_RESULT,
            message_id=message.id,
            payload={'id': call_id, 'name': name, 'preview': observation[:500]},
        )


def _record_finish(result: RolloutResult, step: int) -> None:
    """Note that the conversation reached ACEBench's finish marker."""
    result.trace.add_event(step=step, type=EventType.SUBMIT, payload={'reason': FINISH_MARKER})


def _record_max_steps(result: RolloutResult, step: int) -> None:
    """Note that the rollout ran out of its step budget without finishing."""
    result.trace.add_event(step=step, type=EventType.ERROR, payload={'message': 'max_steps_exceeded'})


def _usage_dict(usage: Optional[ModelUsage]) -> Optional[Dict[str, int]]:
    """Flatten ``ModelUsage`` into the trace's ``input``/``output``/``total`` shape."""
    if usage is None:
        return None
    return {'input': usage.input_tokens, 'output': usage.output_tokens, 'total': usage.total_tokens}


def _execute(message: Any, instances: Dict[str, Any], catch_decode_error: bool) -> _Execution:
    """Decode the agent message into calls, run them, and return the observation."""
    try:
        calls = decode_execution_calls(str(message))
    except Exception as error:
        if not catch_decode_error:
            raise
        # Upstream nudges the multi-step agent back on track with this exact sentence.
        return _Execution(observation=DECODE_ERROR_NUDGE, decode_error=str(error))

    outcomes = [_reparse(_serialize(outcome)) for outcome in _dispatch(calls, instances)]
    return _Execution(observation=outcomes, calls=calls, outcomes=outcomes)


def _dispatch(calls: List[Dict[str, Dict[str, Any]]], instances: Dict[str, Any]) -> List[Any]:
    """Invoke each decoded call and collect one observation per call.

    A call is run on *every* involved instance that exposes the method and the last result is the
    observation, which is what upstream does. It matters: the phone APIs all inherit ``BaseApi``, so
    ``turn_on_wifi()`` has to flip the flag on each of them or later calls see a disconnected device.
    """
    outcomes = []
    for call in calls:
        name, arguments = next(iter(call.items()))
        methods = _resolve_methods(name, instances)
        if not methods:
            # Upstream reuses the previous observation here; reporting the failure is clearer and
            # leaves the graded state unchanged either way.
            outcomes.append(f"Error during execution: name '{name}' is not defined")
            continue
        try:
            outcome = None
            for method in methods:
                outcome = method(**arguments)
            outcomes.append(outcome)
        except Exception as error:  # noqa: BLE001 - surfaced to the agent, as upstream does
            outcomes.append(f'Error during execution: {error}')
    return outcomes


def _resolve_methods(name: str, instances: Dict[str, Any]) -> List[Any]:
    """Collect the public method ``name`` from every involved instance that exposes it."""
    if name.startswith('_'):
        return []
    methods = [getattr(instance, name, None) for instance in instances.values()]
    return [method for method in methods if callable(method)]


def _serialize(outcome: Any) -> str:
    """Render an execution outcome as text, as upstream does before handing it back."""
    if isinstance(outcome, str):
        return outcome
    if isinstance(outcome, dict):
        try:
            return json.dumps(outcome)
        except (TypeError, ValueError):
            return str(outcome)
    return str(outcome)


def _reparse(text: str) -> Any:
    """Decode JSON observations back into objects, leaving anything else as text."""
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return text


def _looks_like_calls(message: str) -> bool:
    """Decide whether the agent addressed the executor, as the official agent does.

    ``APIAgent_turn.respond`` matches ``r'\\[.*\\]'`` without ``re.DOTALL``, so a call whose
    brackets straddle a newline is *not* recognised and the message goes to the user instead. That
    looks like a bug, but widening the pattern here would route messages upstream never executed
    and move the reported scores away from the leaderboard.
    """
    if not re.search(r'\[.*\]', message or ''):
        return False
    try:
        decode_execution_calls(message)
    except Exception:
        return False
    return True


def _is_finished(message: Any) -> bool:
    return FINISH_MARKER in str(message)


def _transcript_line(entry: Tuple[str, Any], execution_label: str) -> str:
    """Render one history entry the way ACEBench appends it to the transcript."""
    sender, message = entry
    label = execution_label if sender == 'execution' else sender
    return f'{label}:{message}\n'


def _strip_role_prefix(text: str) -> str:
    """Drop the ``user:``/``agent:`` prefix before handing a message to the user simulator."""
    for prefix in ('user:', 'agent:'):
        if text.startswith(prefix):
            return text[len(prefix) :]
    return text
