import json
import sys
import tau2.utils.llm_utils as tau_llm_utils
from tau2.data_model.message import AssistantMessage, Message, ToolCall
from tau2.data_model.tasks import Task
from tau2.environment.tool import Tool
from tau2.run import run_task
from tau2.utils.llm_utils import to_litellm_messages
from typing import Any, Callable, Dict, List, Optional, Sequence

from evalscope.api.dataset.dataset import Sample
from evalscope.api.messages.chat_message import dict_to_chat_message
from evalscope.api.model import GenerateConfig, get_model
from evalscope.api.model.model import Model
from evalscope.api.model.model_output import ChatCompletionChoice, ModelOutput
from evalscope.api.tool.tool_info import ToolInfo
from evalscope.constants import EvalType
from evalscope.models.utils.openai import openai_chat_choices
from evalscope.utils.function_utils import run_once

MODEL_DICT: Dict[str, Model] = {
    'agent': None,
    'user': None,
}

_MODEL_PATCHED: bool = False
_ORIGINAL_TAU2_GENERATE: Optional[Callable[..., Any]] = None


def _patch_tau2_generate(new_generate: Callable[..., Any]) -> None:
    """Fan-out monkey patch for Tau2 when consumers did `from ... import generate`."""
    global _MODEL_PATCHED, _ORIGINAL_TAU2_GENERATE
    if _MODEL_PATCHED:
        return

    original = getattr(tau_llm_utils, 'generate', None)
    if original is None:
        raise RuntimeError('tau2.utils.llm_utils.generate not found')

    # Replace on the source module first
    if original is not new_generate:
        tau_llm_utils.generate = new_generate

    # Fan-out to all tau2 submodules that may hold a direct reference
    for mod_name, mod in list(sys.modules.items()):
        if not (isinstance(mod_name, str) and mod_name.startswith('tau2')):
            continue
        mod_obj = sys.modules.get(mod_name)
        if mod_obj is None:
            continue
        try:
            # Common direct binding: `generate` at module top-level
            if getattr(mod_obj, 'generate', None) is original:
                setattr(mod_obj, 'generate', new_generate)
            # Replace any other aliases that equal the original function
            for attr, val in list(vars(mod_obj).items()):
                if val is original:
                    setattr(mod_obj, attr, new_generate)
        except Exception:
            # Best-effort: ignore modules that disallow setattr or have weird loaders
            pass

    _ORIGINAL_TAU2_GENERATE = original
    _MODEL_PATCHED = True


@run_once
def build_model(agent_model, adapter_instance):

    user_server = get_model(
        model=adapter_instance.user_model,
        eval_type=EvalType.SERVICE,
        base_url=adapter_instance.api_base,
        api_key=adapter_instance.api_key,
        config=GenerateConfig(**adapter_instance.generation_config)
    )
    MODEL_DICT['user'] = user_server
    MODEL_DICT['agent'] = agent_model
    # Patch Tau2 generate function for `from ... import generate` consumers
    _patch_tau2_generate(patched_generate)


def patched_generate(
    model: str,
    messages: List[Message],
    tools: Optional[List[Tool]] = None,
    tool_choice: Optional[Any] = None,
    **kwargs: Any,
) -> AssistantMessage:
    """
    Generate a response via an OpenAI-compatible /chat/completions call.

    - Reads EVALSCOPE_API_KEY and EVALSCOPE_BASE_URL from environment.
    - Uses OpenAI chat format for messages/tools/tool_choice.
    - Returns Tau2 AssistantMessage with optional tool_calls and usage.
    """
    global MODEL_DICT

    oa_model = MODEL_DICT.get(model)
    assert oa_model is not None, f'Model {model} not found in MODEL_DICT'

    oa_messages = to_litellm_messages(messages)
    tools = [tool.openai_schema for tool in tools] if tools else None
    if tools and tool_choice is None:
        tool_choice = 'auto'

    # Perform request
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

    return AssistantMessage(
        role='assistant',
        content=msg.content,
        tool_calls=tool_calls,
        cost=None,
        usage=usage,
        raw_data=completion.model_dump(),
    )


def predict(model: Model, sample: Sample, adapter_instance) -> ModelOutput:

    build_model(agent_model=model, adapter_instance=adapter_instance)

    domain = sample.subset_key
    task = Task.model_validate(sample.metadata)
    res = run_task(
        domain=domain,
        task=task,
        agent='llm_agent',
        user='user_simulator',
        llm_agent='agent',
        llm_user='user',
    )

    sample.metadata['task_result'] = res.reward_info.model_dump()
    return ModelOutput(
        model=model.name,
        choices=[ChatCompletionChoice.from_content(res.model_dump_json(indent=2))],
    )
