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
"""
import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from evalscope.api.messages import ChatMessage, ChatMessageAssistant, ChatMessageSystem, ChatMessageUser
from evalscope.api.model import Model, ModelUsage
from evalscope.utils.logger import get_logger
from .parser import decode_execution_calls
from .prompts import agent_prompt_set, build_agent_prompts, build_user_simulator_prompt
from .scenarios import load_scenario_instances, snapshot_states

logger = get_logger()

FINISH_MARKER = 'finish conversation'


@dataclass
class RolloutResult:
    """Outcome of an agent rollout."""

    process: List[str] = field(default_factory=list)
    """Messages the agent sent to the executor, in order, as ACEBench records them."""

    end_state: List[Dict[str, Any]] = field(default_factory=list)
    """Graded attributes of every involved API class after the rollout."""

    messages: List[ChatMessage] = field(default_factory=list)
    """Transcript of the rollout, for inspection in the review files."""

    usage: ModelUsage = field(default_factory=ModelUsage)
    """Token usage accumulated over the rollout."""

    steps: int = 0
    """Number of loop iterations performed."""


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
        The recorded process trace and end state.
    """
    test_category = metadata.get('test_category', '')
    instances = load_scenario_instances(
        initial_config=metadata.get('initial_config') or {},
        involved_classes=metadata.get('involved_classes') or [],
        language=metadata.get('language', 'en'),
    )

    if 'multi_step' in test_category:
        result = _run_multi_step(model, metadata, instances, max_steps)
    else:
        if user_model is None:
            raise ValueError(
                'agent_multi_turn needs a user simulator; set extra_params.user_model to the model '
                'that should play the user (the official runner uses gpt-4o).'
            )
        result = _run_multi_turn(model, user_model, metadata, instances, max_steps)

    result.end_state = snapshot_states(instances)
    return result


def _run_multi_step(
    model: Model,
    metadata: Dict[str, Any],
    instances: Dict[str, Any],
    max_steps: int,
) -> RolloutResult:
    """Drive the agent-executor loop of ``agent_multi_step``."""
    result = RolloutResult()
    transcript = ''
    # ACEBench seeds the history with the task description as a user message.
    history: List[Tuple[str, Any]] = [('user', metadata.get('question', ''))]

    for step in range(max_steps):
        last_sender = history[-1][0]
        if step == 0 or last_sender == 'execution':
            # Upstream only folds user and execution messages into the transcript here, which means
            # the agent never sees its own previous messages in this category.
            transcript += _transcript_line(history[-1], execution_label='execution result')
            message = _agent_step(model, metadata, transcript, result)
            history.append(('agent', message))
        else:
            result.process.append(history[-1][1])
            observation = _execute(history[-1][1], instances, catch_decode_error=True)
            history.append(('execution', observation))

        result.steps = step + 1
        if step > 1 and _is_finished(history[-1][1]):
            break

    result.messages = _to_chat_messages(history)
    return result


def _run_multi_turn(
    model: Model,
    user_model: Model,
    metadata: Dict[str, Any],
    instances: Dict[str, Any],
    max_steps: int,
) -> RolloutResult:
    """Drive the user-agent-executor loop of ``agent_multi_turn``."""
    result = RolloutResult()
    language = metadata.get('language', 'en')
    involved_classes = metadata.get('involved_classes') or []
    templates = agent_prompt_set(language)

    simulator = _UserSimulator(
        model=user_model,
        system_prompt=build_user_simulator_prompt(metadata.get('question', ''), involved_classes, language),
        opening=templates['user_opening'],
    )

    transcript = ''
    history: List[Tuple[str, Any]] = [('user', simulator.start(result))]
    # Recipient of the last message, which is what upstream dispatches on.
    recipient = 'agent'

    for step in range(max_steps):
        transcript += _transcript_line(history[-1], execution_label='execution')
        if recipient == 'user':
            message = simulator.respond(history[-1][1], result)
            history.append(('user', message))
            recipient = 'agent'
        elif recipient == 'agent':
            message = _agent_step(model, metadata, transcript, result)
            history.append(('agent', message))
            recipient = 'execution' if _looks_like_calls(message) else 'user'
        else:
            result.process.append(history[-1][1])
            observation = _execute(history[-1][1], instances, catch_decode_error=False)
            history.append(('execution', observation))
            recipient = 'agent'

        result.steps = step + 1
        if step > 1 and _is_finished(history[-1][1]):
            break

    result.messages = _to_chat_messages(history)
    return result


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

    def start(self, result: RolloutResult) -> str:
        """Produce the opening user message."""
        return self._generate(result)

    def respond(self, agent_message: Any, result: RolloutResult) -> str:
        """Answer the agent's latest message."""
        self.messages.append(ChatMessageUser(content=_strip_role_prefix(str(agent_message))))
        return self._generate(result)

    def _generate(self, result: RolloutResult) -> str:
        output = self.model.generate(self.messages)
        if output.usage is not None:
            result.usage += output.usage
        response = output.completion or ''
        self.messages.append(ChatMessageSystem(content=response))
        return response


def _agent_step(model: Model, metadata: Dict[str, Any], transcript: str, result: RolloutResult) -> str:
    """Ask the model under evaluation for its next message."""
    system_prompt, user_prompt = build_agent_prompts(
        transcript=transcript,
        functions=metadata.get('functions') or [],
        involved_classes=metadata.get('involved_classes') or [],
        test_category=metadata.get('test_category', ''),
        language=metadata.get('language', 'en'),
    )
    output = model.generate([ChatMessageSystem(content=system_prompt), ChatMessageUser(content=user_prompt)])
    if output.usage is not None:
        result.usage += output.usage
    return output.completion or ''


def _execute(message: Any, instances: Dict[str, Any], catch_decode_error: bool) -> Any:
    """Decode the agent message into calls, run them, and return the observation."""
    try:
        calls = decode_execution_calls(str(message))
    except Exception:
        if not catch_decode_error:
            raise
        # Upstream nudges the multi-step agent back on track with this exact sentence.
        return 'Please do not ask me any questions, use the known conditions to solve the problem'

    return [_reparse(_serialize(outcome)) for outcome in _dispatch(calls, instances)]


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
    """Decide whether the agent addressed the executor, as the official agent does."""
    if not re.search(r'\[.*\]', message or '', re.DOTALL):
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
            return text[len(prefix):]
    return text


def _to_chat_messages(history: List[Tuple[str, Any]]) -> List[ChatMessage]:
    """Render the rollout history as chat messages for the review files."""
    messages: List[ChatMessage] = []
    for sender, message in history:
        content = message if isinstance(message, str) else str(message)
        if sender == 'agent':
            messages.append(ChatMessageAssistant(content=content))
        else:
            messages.append(ChatMessageUser(content=f'{sender}: {content}'))
    return messages
