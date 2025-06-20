import json
import time
import uuid
from typing import Any, List, Optional, Union

from evalscope.utils.logger import get_logger
from .server_adapter import ServerModelAdapter

logger = get_logger()


class BFCLAdapter(ServerModelAdapter):
    """
    BFCL model adapter to request remote API model and generate results for BFCL evaluation.
    Support multi-turn and single-turn function calling tasks.
    """

    def __init__(self, api_url: str, model_id: str, api_key: str = 'EMPTY', **kwargs):
        """
        Args:
            api_url: The URL of the remote API model.
            model_id: The ID of the remote API model.
            api_key: The API key of the remote API model.
        """
        super().__init__(api_url=api_url, model_id=model_id, api_key=api_key, **kwargs)

    def predict(self, inputs: List[dict], infer_cfg: Optional[dict] = None) -> List[dict]:
        """
        Model prediction func. For multi-turn evals, we pass a list[list[message]] to the model
        where each list is a follow up turn in the conversation
        each turn is a List[List[Message]]

        Args:
            inputs (List[dict]): The input data.
            infer_cfg (dict): Inference configuration.

        Returns:
            res (List[dict]): The model prediction results.
        """
        infer_cfg = infer_cfg or {}
        results = []

        for input_item in inputs:
            # This flag decides if we pass tools to the API or try tool calling via prompting
            # Passing tools to the API means that we rely on the API to manage system prompt specifics
            # and also expect parsed tool calls in the ChatCompletionMessage object
            # This is how the is_fc_model=True benchmark is designed to work
            # On the other hand, we try to manage
            # tool calling via prompting and parse tool calls in the standard text response
            # This is how the is_fc_model=False benchmark is designed to work
            row = input_item.get('messages')
            is_fc_model = row.get('is_fc_model', False)

            if is_fc_model:
                response = self.generate_turn_with_tools(row, infer_cfg)
            else:
                response = self.generate_turn(row, infer_cfg)

            # wrap response with openai types
            res_d = {
                'choices': [{
                    'index': 0,
                    'message': {
                        'content': response,
                        'role': 'assistant'
                    }
                }],
                'created': time.time(),
                'model': self.model_id,
                'object': 'chat.completion',
                'usage': {
                    'completion_tokens': 0,
                    'prompt_tokens': 0,
                    'total_tokens': 0
                }
            }
            results.append(res_d)

        return results

    def generate_turn(self, row: dict[str, Any], infer_cfg: dict[str, Any]) -> list[str]:
        from bfcl_eval.constants.default_prompts import (DEFAULT_USER_PROMPT_FOR_ADDITIONAL_FUNCTION_PROMPTING,
                                                         MAXIMUM_STEP_LIMIT)
        from bfcl_eval.eval_checker.multi_turn_eval.multi_turn_utils import execute_multi_turn_func_call
        from bfcl_eval.model_handler.utils import default_decode_execute_prompting

        all_model_responses = []
        current_messages = []
        turns = row['turns']
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
                        functions=row['missing_functions'][str(turn_idx)]),
                }]
                current_messages += new_turn

            while True:
                input_item = {
                    'messages': current_messages,
                }
                responses = self.process_single_input(input_item, infer_cfg)
                result = responses['choices'][0]['message']['content']

                logger.debug(f'Turn:{turn_idx} Step:{n_steps} Result: {result}')
                current_messages.append({
                    'role': 'assistant',
                    'content': result,
                })
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
                    logger.error(f'INFERENCE_ERROR: Exceeded max inference steps ({MAXIMUM_STEP_LIMIT})')
                    break

            all_model_responses.append(current_responses)

        return all_model_responses

    def generate_turn_with_tools(self, row: dict[str, Any], infer_cfg: dict[str, Any]) -> list[str]:
        from bfcl_eval.constants.default_prompts import (DEFAULT_USER_PROMPT_FOR_ADDITIONAL_FUNCTION_FC,
                                                         MAXIMUM_STEP_LIMIT)
        from bfcl_eval.eval_checker.multi_turn_eval.multi_turn_utils import execute_multi_turn_func_call
        from bfcl_eval.model_handler.utils import convert_to_function_call

        all_model_responses = []
        current_messages = []
        turns = row['turns']
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
                    tools.append({
                        'type': 'function',
                        'function': new_tool[0],
                    })
                new_turn = [{
                    'role': 'user',
                    'content': DEFAULT_USER_PROMPT_FOR_ADDITIONAL_FUNCTION_FC,
                }]
                current_messages += new_turn

            while True:
                input_item = {
                    'messages': current_messages,
                    'tools': tools,
                }
                responses = self.process_single_input(input_item, infer_cfg)
                message = responses['choices'][0]['message']

                current_messages.append(message)
                if isinstance(message, str):
                    model_responses = [message]
                    tool_call_strs = None
                elif message.get('tool_calls'):
                    model_responses = [{
                        tc['function']['name']: tc['function']['arguments']
                    } for tc in message['tool_calls']]
                    try:
                        tool_call_strs = convert_to_function_call(model_responses)
                    except Exception as e:
                        logger.error(f'Error converting tool calls to function call strings: {e}')
                        tool_call_strs = None
                else:
                    model_responses = [message['content']]
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

                    for tc, tool_output in zip(message['tool_calls'], tool_outputs, strict=False):
                        current_messages.append({
                            'role': 'tool',
                            'tool_call_id': tc['id'],
                            'content': json.dumps({'response': tool_output}),
                        })
                else:
                    break

                n_steps += 1
                if n_steps > MAXIMUM_STEP_LIMIT:
                    logger.error(f'INFERENCE_ERROR: Exceeded max inference steps ({MAXIMUM_STEP_LIMIT})')
                    break

            all_model_responses.append(current_responses)

        return all_model_responses
