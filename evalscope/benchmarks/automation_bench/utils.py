import atexit
import importlib
import json
import signal
import sys
import threading
from contextlib import contextmanager
from typing import Any, Dict, Iterable, Iterator, List

from evalscope.api.messages import ChatMessage, dict_to_chat_message
from evalscope.utils.function_utils import AsyncioLoopRunner

DEFAULT_AUTOMATION_BENCH_COMMIT = 'a321764ace3cfbe42289e6a13abef2f0f4f56fad'
DEFAULT_AUTOMATION_BENCH_PACKAGE = (
    f'automation-bench @ git+https://github.com/zapier/AutomationBench.git@{DEFAULT_AUTOMATION_BENCH_COMMIT}'
)
DEFAULT_AUTOMATION_BENCH_VERIFIERS = 'verifiers==0.1.12.dev2'
PUBLIC_DOMAINS = ['sales', 'marketing', 'operations', 'support', 'finance', 'hr']
_ENVIRONMENT_INIT_LOCK = threading.Lock()


def ensure_automation_bench_runtime() -> None:
    """Ensure the pinned official AutomationBench package is available."""
    if sys.version_info < (3, 13):
        raise RuntimeError(
            'AutomationBench requires Python 3.13+. Create and activate a Python 3.13 conda environment, '
            'then run EvalScope from that environment.'
        )

    try:
        importlib.import_module('automationbench.runner')
    except ImportError as error:
        raise ImportError(
            'AutomationBench is not installed in the active Python environment. Install the pinned official '
            f'package with `python -m pip install "{DEFAULT_AUTOMATION_BENCH_VERIFIERS}" '
            f'"{DEFAULT_AUTOMATION_BENCH_PACKAGE}"`, then rerun EvalScope.'
        ) from error


def load_automation_bench_records(domains: Iterable[str]) -> Dict[str, List[Dict[str, Any]]]:
    """Load task records from the official domain datasets."""
    from automationbench.domains import get_domain_dataset

    return {domain: [dict(row) for row in get_domain_dataset(domain)] for domain in domains}


def run_automation_bench_task(
    task_record: Dict[str, Any],
    model_name: str,
    model_api: Any,
    api: str,
    toolset: str,
    max_turns: int,
    sampling_args: Dict[str, Any],
    extra_headers: Dict[str, str],
) -> Dict[str, Any]:
    """Evaluate one task with the official environment on EvalScope's model client."""
    return AsyncioLoopRunner.run(
        _run_automation_bench_task(
            task_record=task_record,
            model_name=model_name,
            model_api=model_api,
            api=api,
            toolset=toolset,
            max_turns=max_turns,
            sampling_args=sampling_args,
            extra_headers=extra_headers,
        )
    )


async def _run_automation_bench_task(
    task_record: Dict[str, Any],
    model_name: str,
    model_api: Any,
    api: str,
    toolset: str,
    max_turns: int,
    sampling_args: Dict[str, Any],
    extra_headers: Dict[str, str],
) -> Dict[str, Any]:
    from automationbench.clients import (
        OpenAIResponsesClient,
        RetryingOpenAIChatCompletionsClient,
        StreamingAnthropicClient,
    )
    from automationbench.rubric import create_rubric
    from automationbench.runner import AutomationBenchEnv
    from datasets import Dataset

    try:
        native_client = model_api.async_client
    except AttributeError as error:
        raise TypeError(
            'AutomationBench requires an API model with an async client. Use openai_api, openai_responses_api, '
            'or anthropic_api evaluation type.'
        ) from error
    if extra_headers:
        native_client = native_client.with_options(default_headers=extra_headers)

    if api == 'anthropic':
        client = StreamingAnthropicClient(native_client)
    elif api == 'responses':
        client = OpenAIResponsesClient(native_client)
    else:
        client = RetryingOpenAIChatCompletionsClient(native_client)

    with _official_environment_init_context():
        env = AutomationBenchEnv(
            dataset=Dataset.from_list([task_record]),
            rubric=create_rubric(),
            max_turns=max_turns,
            toolset=toolset,
            search_top_k=20,
        )

    results = await env.evaluate(
        client=client,
        model=model_name,
        sampling_args=sampling_args or None,
        num_examples=1,
        rollouts_per_example=1,
        max_concurrent=1,
        state_columns=['_usage', '_debug', '_assertion_results', '_end_state', '_perf'],
    )
    outputs = results.get('outputs') if isinstance(results, dict) else None
    if not isinstance(outputs, list) or len(outputs) != 1:
        raise RuntimeError(
            f'AutomationBench returned {len(outputs) if isinstance(outputs, list) else 0} outputs for one task.'
        )
    return _normalize_result(outputs[0])


@contextmanager
def _official_environment_init_context() -> Iterator[None]:
    """Suppress Verifiers' process hooks while constructing an environment in a worker thread."""
    if threading.current_thread() is threading.main_thread():
        yield
        return

    # Verifiers 0.1.12.dev2 unconditionally installs signal and atexit handlers in Environment.__post_init__.
    # EvalScope constructs each sample environment in a worker thread, where signal.signal raises ValueError. The
    # upstream API has no switch for these process hooks, so suppress only the two registrations during construction.
    with _ENVIRONMENT_INIT_LOCK:
        original_signal = signal.signal
        original_atexit_register = atexit.register

        def ignore_registration(*args: Any, **kwargs: Any) -> None:
            return None

        signal.signal = ignore_registration
        atexit.register = ignore_registration
        try:
            yield
        finally:
            signal.signal = original_signal
            atexit.register = original_atexit_register


def _normalize_result(raw_output: Dict[str, Any]) -> Dict[str, Any]:
    raw = dict(raw_output)
    metrics = dict(raw.get('metrics') or {})
    partial_credit = float(metrics.get('partial_credit', raw.get('reward', 0.0)) or 0.0)
    completed = float(metrics.get('task_completed_correctly', partial_credit == 1.0) or 0.0)
    debug = dict(raw.get('_debug') or {})
    errors = debug.get('errors') or []
    error = raw.get('error') or (str(errors[0]) if errors else None)
    prompt = [_serialize_message(message) for message in (raw.get('prompt') or [])]
    completion = [_serialize_message(message) for message in (raw.get('completion') or [])]
    return {
        'task': raw.get('task', 'unknown'),
        'metrics': {
            'partial_credit': partial_credit,
            'task_completed_correctly': completed,
        },
        'messages': prompt + completion,
        'usage': raw.get('_usage') or {},
        'debug': debug,
        'assertion_results': raw.get('_assertion_results') or [],
        'end_state': raw.get('_end_state'),
        'perf': raw.get('_perf') or {},
        'error': error,
    }


def convert_automation_bench_messages(messages: Iterable[Any]) -> List[ChatMessage]:
    """Convert official Verifiers messages into EvalScope chat messages."""
    converted: List[ChatMessage] = []
    for raw_message in messages:
        data = _serialize_message(raw_message)
        role = data.get('role')
        if role not in ('system', 'user', 'assistant', 'tool'):
            continue

        data['content'] = _message_content(data.get('content'))
        if role == 'assistant':
            data['tool_calls'] = _normalize_tool_calls(data.get('tool_calls')) or None
            reasoning = data.pop('reasoning_content', None)
            if reasoning:
                data['reasoning'] = reasoning

        try:
            message = dict_to_chat_message(data)
        except ValueError:
            if role != 'assistant' or not data.get('tool_calls'):
                raise
            data['metadata'] = {**(data.get('metadata') or {}), 'unparsed_tool_calls': data.pop('tool_calls')}
            message = dict_to_chat_message(data)
        message.source = 'input' if role in ('system', 'user') else 'generate'
        converted.append(message)
    return converted


def _normalize_tool_calls(tool_calls: Any) -> List[Any]:
    if not tool_calls:
        return []
    raw_calls = tool_calls if isinstance(tool_calls, list) else [tool_calls]
    normalized = []
    for raw_call in raw_calls:
        if isinstance(raw_call, str):
            try:
                call = json.loads(raw_call)
            except json.JSONDecodeError:
                normalized.append(raw_call)
                continue
        else:
            call = _serialize_message(raw_call)
        if isinstance(call, dict) and not isinstance(call.get('function'), dict):
            call['function'] = {
                'name': call.pop('name', ''),
                'arguments': call.pop('arguments', '{}'),
            }
        normalized.append(call)
    return normalized


def _serialize_message(message: Any) -> Dict[str, Any]:
    if isinstance(message, dict):
        return dict(message)
    return message.model_dump(mode='json')


def _message_content(content: Any) -> str:
    if content is None:
        return ''
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
                continue
            data = _serialize_message(item)
            parts.append(str(data.get('text') or data.get('content') or data))
        return '\n'.join(parts)
    return str(content)
