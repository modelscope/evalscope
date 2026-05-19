import json
import sys
import tau2.utils.llm_utils as tau_llm_utils
from tau2.data_model.message import AssistantMessage, Message, ToolCall
from tau2.data_model.tasks import Task
from tau2.environment.tool import Tool
from tau2.evaluator.evaluator import EvaluationType
from tau2.run import run_task
from tau2.utils.llm_utils import to_litellm_messages
from typing import Any, Callable, Dict, List, Optional

from evalscope.api.dataset.dataset import Sample
from evalscope.api.evaluator import InferenceResult
from evalscope.api.messages.chat_message import dict_to_chat_message
from evalscope.api.model import GenerateConfig, get_model
from evalscope.api.model.model import Model
from evalscope.api.model.model_output import ChatCompletionChoice, ModelOutput
from evalscope.api.tool.tool_info import ToolInfo
from evalscope.constants import EvalType
from evalscope.models.utils.openai import openai_chat_choices
from evalscope.utils.function_utils import run_once

KNOWLEDGE_DOMAIN = 'banking_knowledge'

MODEL_DICT: Dict[str, Model] = {
    'agent': None,
    'user': None,
}

_MODEL_PATCHED: bool = False
_ORIGINAL_TAU2_GENERATE: Optional[Callable[..., Any]] = None


def _patch_tau2_generate(new_generate: Callable[..., Any]) -> None:
    """Fan-out monkey patch for Tau2/Tau3 when consumers did `from ... import generate`."""
    global _MODEL_PATCHED, _ORIGINAL_TAU2_GENERATE
    if _MODEL_PATCHED:
        return

    original = getattr(tau_llm_utils, 'generate', None)
    if original is None:
        raise RuntimeError('tau2.utils.llm_utils.generate not found')

    if original is not new_generate:
        tau_llm_utils.generate = new_generate

    for mod_name, mod in list(sys.modules.items()):
        if not (isinstance(mod_name, str) and mod_name.startswith('tau2')):
            continue
        mod_obj = sys.modules.get(mod_name)
        if mod_obj is None:
            continue
        try:
            if getattr(mod_obj, 'generate', None) is original:
                setattr(mod_obj, 'generate', new_generate)
            for attr, val in list(vars(mod_obj).items()):
                if val is original:
                    setattr(mod_obj, attr, new_generate)
        except Exception:
            pass

    _ORIGINAL_TAU2_GENERATE = original
    _MODEL_PATCHED = True


@run_once
def build_model(agent_model, adapter_instance):

    user_server = get_model(
        model=adapter_instance.user_model,
        eval_type=EvalType.OPENAI_API,
        base_url=adapter_instance.api_base,
        api_key=adapter_instance.api_key,
        config=GenerateConfig(**adapter_instance.generation_config)
    )
    MODEL_DICT['user'] = user_server
    MODEL_DICT['agent'] = agent_model
    _patch_tau2_generate(patched_generate)


def patched_generate(
    model: str,
    messages: List[Message],
    tools: Optional[List[Tool]] = None,
    tool_choice: Optional[Any] = None,
    **kwargs: Any,
) -> AssistantMessage:
    """Generate via evalscope models, returning a Tau2 AssistantMessage."""
    global MODEL_DICT

    # tau2 internal callers may request other model names (e.g. the NL-assertion
    # judge uses 'gpt-4.1-2025-04-14'). Route any unknown name to the user model
    # so judging still works without requiring an additional API key. This keeps
    # the integration self-contained at the cost of using the user_model as judge.
    oa_model = MODEL_DICT.get(model) or MODEL_DICT.get('user')
    assert oa_model is not None, f'Model {model} not found in MODEL_DICT'

    oa_messages = to_litellm_messages(messages)
    tools = [tool.openai_schema for tool in tools] if tools else None
    if tools and tool_choice is None:
        tool_choice = 'auto'

    completion = oa_model.generate(
        input=[dict_to_chat_message(msg) for msg in oa_messages],
        tools=[ToolInfo.model_validate(tool['function']) for tool in tools] if tools else None,
        tool_choice=tool_choice,
    )

    oa_choices = openai_chat_choices(completion.choices, include_reasoning=False)
    choice = oa_choices[0]
    msg = choice.message

    tool_calls = msg.tool_calls or []
    tool_calls = [
        ToolCall(
            id=tool_call.id,
            name=tool_call.function.name,
            arguments=json.loads(tool_call.function.arguments),
        ) for tool_call in tool_calls
    ]
    tool_calls = tool_calls or None
    usage = completion.usage.model_dump(exclude_none=True)

    raw_data = completion.model_dump()
    perf = completion.message.perf_metrics
    if perf is not None:
        raw_data['_perf_metrics'] = perf.model_dump()

    return AssistantMessage(
        role='assistant',
        content=msg.content,
        tool_calls=tool_calls,
        cost=None,
        usage=usage,
        raw_data=raw_data,
    )


def predict(model: Model, sample: Sample, adapter_instance) -> InferenceResult:

    build_model(agent_model=model, adapter_instance=adapter_instance)

    domain = sample.subset_key
    task_data = {k: v for k, v in sample.metadata.items() if k != '_domain'}
    task = Task.model_validate(task_data)

    run_task_kwargs = dict(
        domain=domain,
        task=task,
        agent='llm_agent',
        user='user_simulator',
        llm_agent='agent',
        llm_user='user',
        # ALL_WITH_NL_ASSERTIONS so retail-style tasks whose reward_basis
        # includes NL assertions can be scored. NL judging routes through our
        # patched_generate fallback (uses the user_model as judge).
        evaluation_type=EvaluationType.ALL_WITH_NL_ASSERTIONS,
    )
    # retrieval_config only applies to the knowledge domain
    if domain == KNOWLEDGE_DOMAIN:
        run_task_kwargs['retrieval_config'] = adapter_instance.retrieval_config
        run_task_kwargs['retrieval_config_kwargs'] = adapter_instance.retrieval_config_kwargs

    # tau2 v1.0.0 raises on a few known evaluator edge cases (hallucinated tool
    # names during replay, NL-assertion reward basis without ALL_WITH_NL_ASSERTIONS, ...).
    # Treat these as task failures (reward=0) instead of aborting the whole run.
    try:
        res = run_task(**run_task_kwargs)
    except Exception as e:
        from loguru import logger as _logger
        _logger.warning(f'tau2.run_task failed for domain={domain} task_id={task.id}: {e}')
        sample.metadata['task_result'] = {'reward': 0.0, 'error': str(e)}
        output = ModelOutput(
            model=model.name,
            choices=[ChatCompletionChoice.from_content(f'tau2 run_task error: {e}')],
        )
        return InferenceResult(output=output, messages=None)

    sample.metadata['task_result'] = res.reward_info.model_dump()

    from evalscope.api.messages.chat_message import ChatMessageAssistant
    from evalscope.api.messages.perf_metrics import PerformanceMetrics

    raw_msgs = res.messages or []
    li_msgs = to_litellm_messages(raw_msgs)
    agent_messages = []
    for raw, li in zip(raw_msgs, li_msgs):
        try:
            chat_msg = dict_to_chat_message(li)
        except Exception:
            continue
        if isinstance(chat_msg, ChatMessageAssistant) and isinstance(raw, AssistantMessage):
            perf = (raw.raw_data or {}).get('_perf_metrics') if raw.raw_data else None
            if perf:
                chat_msg.perf_metrics = PerformanceMetrics(**perf)
        agent_messages.append(chat_msg)

    output = ModelOutput(
        model=model.name,
        choices=[ChatCompletionChoice.from_content(res.model_dump_json(indent=2))],
    )
    return InferenceResult(output=output, messages=agent_messages or None)
