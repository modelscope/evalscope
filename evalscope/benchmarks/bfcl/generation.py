import json
import time
from typing import Any

from evalscope.api.dataset import Sample
from evalscope.api.messages import dict_to_chat_message
from evalscope.api.model import ChatCompletionChoice, Model, ModelOutput, ModelUsage
from evalscope.api.tool.tool_info import ToolInfo
from evalscope.utils.logger import get_logger

logger = get_logger()


def predict(model: Model, sample: Sample) -> ModelOutput:
    """Main prediction function for BFCL using the new API framework."""
    # Extract the row data from sample metadata
    row = sample.metadata
    is_fc_model = row.get('is_fc_model', False)

    if is_fc_model:
        response, model_usage = generate_turn_with_tools(model, row)
    else:
        response, model_usage = generate_turn(model, row)

    sample.metadata['generation'] = response
    # wrap response with openai types
    return ModelOutput(
        model=model.name,
        choices=[ChatCompletionChoice.from_content(json.dumps(response, ensure_ascii=False, indent=2))],
        model_usage=model_usage,
        time=time.time()
    )


def generate_turn(model: Model, row: dict[str, Any]):
    from bfcl_eval.constants.default_prompts import (
        DEFAULT_USER_PROMPT_FOR_ADDITIONAL_FUNCTION_PROMPTING,
        MAXIMUM_STEP_LIMIT,
    )
    from bfcl_eval.eval_checker.multi_turn_eval.multi_turn_utils import execute_multi_turn_func_call
    from bfcl_eval.model_handler.utils import default_decode_execute_prompting

    all_model_responses = []
    current_messages = []
    turns = row['turns']
    model_usage = ModelUsage()

    for turn_idx, messages in enumerate(turns):
        n_steps = 0
        current_responses = []
        current_messages += messages.copy()

        if str(turn_idx) in row['missing_functions']:
            assert len(messages) == 0, 'Holdout turn should not have user message.'
            new_turn = [{
                'role':
                'user',
                'content':
                DEFAULT_USER_PROMPT_FOR_ADDITIONAL_FUNCTION_PROMPTING.format(
                    functions=row['missing_functions'][str(turn_idx)]
                ),
            }]
            current_messages += new_turn

        while True:
            # Create a sample for the current messages
            from evalscope.api.messages.chat_message import dict_to_chat_message
            chat_messages = [dict_to_chat_message(msg) for msg in current_messages]

            # Get model response using generate method
            model_output = model.generate(chat_messages)

            # Handle the response based on the model output structure
            message = model_output.message
            if model_output.usage is not None:
                model_usage += model_output.usage

            current_messages.append(message)
            if isinstance(message, str):
                result = message
            else:
                result = message.text

            logger.debug(f'Turn:{turn_idx} Step:{n_steps} Result: {result}')
            current_responses.append(result)

            execute_tools = row.get('should_execute_tool_calls', False)
            if execute_tools:
                try:
                    tool_calls = default_decode_execute_prompting(result)
                except Exception:
                    tool_calls = None

                if tool_calls is None:
                    break

                tool_outputs, _ = execute_multi_turn_func_call(
                    tool_calls,
                    initial_config=row['initial_config'],
                    involved_classes=row['involved_classes'],
                    model_name='evaluator_loop',
                    test_entry_id=row['id'],
                    long_context=('long_context' in row['test_category'] or 'composite' in row['test_category']),
                    is_evaL_run=False,
                )
                # Append tool outputs to the current messages
                tool_results = []
                for tool_output, tool_call in zip(tool_outputs, tool_calls):
                    tool_results.append({'role': 'tool', 'name': tool_call, 'content': tool_output})
                current_messages.append({
                    'role': 'user',
                    'content': repr(tool_results),
                })
            else:
                break

            n_steps += 1
            if n_steps > MAXIMUM_STEP_LIMIT:
                logger.warning(f'INFERENCE_WARNING: Exceeded max inference steps ({MAXIMUM_STEP_LIMIT})')
                break

        all_model_responses.append(current_responses)

    return all_model_responses, model_usage


def generate_turn_with_tools(model: Model, row: dict[str, Any]):
    from bfcl_eval.constants.default_prompts import DEFAULT_USER_PROMPT_FOR_ADDITIONAL_FUNCTION_FC, MAXIMUM_STEP_LIMIT
    from bfcl_eval.eval_checker.multi_turn_eval.multi_turn_utils import execute_multi_turn_func_call
    from bfcl_eval.model_handler.utils import convert_to_function_call

    all_model_responses = []
    current_messages = []
    turns = row['turns']
    model_usage = ModelUsage()

    for turn_idx, messages in enumerate(turns):
        n_steps = 0
        current_responses = []
        current_messages += messages.copy()
        tools = row['tools']

        if str(turn_idx) in row['missing_functions']:
            assert len(messages) == 0, 'Holdout turn should not have user message.'
            # inject new functions on the fly
            new_tools = row['missing_functions'][str(turn_idx)]
            for new_tool in new_tools:
                cur_tool = new_tool[0]
                cur_tool['parameters']['type'] = 'object'
                tools.append({
                    'type': 'function',
                    'function': cur_tool,
                })
            new_turn = [{
                'role': 'user',
                'content': DEFAULT_USER_PROMPT_FOR_ADDITIONAL_FUNCTION_FC,
            }]
            current_messages += new_turn

        while True:
            # Create a sample for the current messages with tools
            chat_messages = [dict_to_chat_message(msg) for msg in current_messages]
            current_sample = Sample(
                input=chat_messages,
                target='',
                tools=[ToolInfo.model_validate(tool['function']) for tool in tools],
            )

            # Get model response
            model_output = model.generate(current_sample.input, tools=current_sample.tools)

            # Handle the response based on the model output structure
            message = model_output.message
            if model_output.usage is not None:
                model_usage += model_output.usage

            current_messages.append(message)
            if isinstance(message, str):
                model_responses = [message]
                tool_call_strs = None
            elif message.tool_calls:
                model_responses = [{tc.function.name: tc.function.arguments} for tc in message.tool_calls]
                try:
                    tool_call_strs = convert_to_function_call(model_responses)
                except Exception as e:
                    logger.error(f'Error converting tool calls to function call strings: {e}')
                    tool_call_strs = None
            else:
                model_responses = [message.text]
                tool_call_strs = None

            current_responses.extend(model_responses)

            execute_tools = row.get('should_execute_tool_calls', False)
            if execute_tools and tool_call_strs is not None:
                tool_outputs, _ = execute_multi_turn_func_call(
                    tool_call_strs,
                    initial_config=row['initial_config'],
                    involved_classes=row['involved_classes'],
                    model_name='evaluator_loop',
                    test_entry_id=row['id'],
                    long_context=('long_context' in row['test_category'] or 'composite' in row['test_category']),
                    is_evaL_run=False,
                )

                for tc, tool_output in zip(message.tool_calls, tool_outputs, strict=False):
                    current_messages.append({
                        'role': 'tool',
                        'tool_call_id': tc.id,
                        'content': json.dumps({'response': tool_output}),
                    })
            else:
                break

            n_steps += 1
            if n_steps > MAXIMUM_STEP_LIMIT:
                logger.warning(f'INFERENCE_WARNING: Exceeded max inference steps ({MAXIMUM_STEP_LIMIT})')
                break

        all_model_responses.append(current_responses)

    return all_model_responses, model_usage
