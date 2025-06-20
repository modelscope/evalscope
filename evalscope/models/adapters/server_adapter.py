import copy
import openai
from collections import defaultdict
from openai.types.chat import ChatCompletion, ChatCompletionChunk
from openai.types.chat.chat_completion import ChatCompletionMessage, Choice
from typing import List, Optional, Union

from evalscope.utils.logger import get_logger
from evalscope.utils.utils import get_supported_params
from .base_adapter import BaseModelAdapter

logger = get_logger()


class ServerModelAdapter(BaseModelAdapter):
    """
    Server model adapter to request remote API model and generate results.
    """

    def __init__(self, api_url: str, model_id: str, api_key: str = 'EMPTY', **kwargs):
        """
        Args:
            api_url: The URL of the remote API model.
            model_id: The ID of the remote API model.
            api_key: The API key of the remote API model.
        """
        self.api_url = api_url.rstrip('/').rsplit('/chat/completions', 1)[0]
        self.model_id = model_id
        self.api_key = api_key

        self.client = openai.OpenAI(
            api_key=api_key,
            base_url=self.api_url,
        )
        self.supported_params = get_supported_params(self.client.chat.completions.create)

        self.seed = kwargs.get('seed', None)
        self.timeout = kwargs.get('timeout', 60)
        self.stream = kwargs.get('stream', False)
        self.model_cfg = {'api_url': api_url, 'model_id': model_id, 'api_key': api_key}
        super().__init__(model=None, model_cfg=self.model_cfg, **kwargs)

    def predict(self, inputs: List[dict], infer_cfg: Optional[dict] = None) -> List[dict]:
        """
        Model prediction func.

        Args:
            inputs (List[dict]): The input data.
            infer_cfg (dict): Inference configuration.

        Returns:
            res (List[dict]): The model prediction results.
        """
        infer_cfg = infer_cfg or {}
        results = []

        for input_item in inputs:
            response = self.process_single_input(input_item, infer_cfg)
            results.append(response)

        return results

    def process_single_input(self, input_item: dict, infer_cfg: dict) -> dict:
        """Process a single input item."""
        request_json = self.make_request(input_item, infer_cfg)
        response = self.send_request(request_json)
        return response

    def make_request_messages(self, input_item: dict) -> list:
        """
        Make request messages for OpenAI API.
        """
        if input_item.get('messages', None):
            return input_item['messages']

        data: list = input_item['data']
        if isinstance(data[0], tuple):  # for truthful_qa and hellaswag
            query = '\n'.join(''.join(item) for item in data)
            system_prompt = input_item.get('system_prompt', None)
        else:
            query = data[0]
            system_prompt = input_item.get('system_prompt', None)

        messages = []
        if system_prompt:
            messages.append({'role': 'system', 'content': system_prompt})

        messages.append({'role': 'user', 'content': query})

        return messages

    def make_request(self, input_item: dict, infer_cfg: dict) -> dict:
        """Make request to remote API."""
        messages = self.make_request_messages(input_item)
        # Format request JSON according to OpenAI API format
        request_json = {'model': self.model_id, 'messages': messages, **infer_cfg}

        if self.timeout:
            request_json['timeout'] = self.timeout

        request_json['stream'] = self.stream
        if self.stream:
            request_json['stream_options'] = {'include_usage': True}

        if input_item.get('tools', None):
            tools_copy = copy.deepcopy(input_item.get('tools'))
            # Remove the "responses" from the functions, as that doesn't
            # need to go to the model
            for tool in tools_copy:
                if 'function' in tool and 'response' in tool['function']:
                    del tool['function']['response']
            request_json['tools'] = tools_copy

        logger.debug(f'Request to remote API: {request_json}')

        return request_json

    def _parse_extra_params(self, request_json):
        api_params = {}
        extra_body = {}
        for key, value in request_json.items():
            if key in self.supported_params:
                api_params[key] = value
            else:
                extra_body[key] = value

        if extra_body:
            api_params['extra_body'] = extra_body
        return api_params

    def send_request(self, request_json: dict) -> dict:
        try:
            parsed_request = self._parse_extra_params(request_json)
            response = self.client.chat.completions.create(**parsed_request)

            if response and self.stream:
                response = self._collect_stream_response(response)

            return response.model_dump(exclude_unset=True)
        except Exception as e:
            logger.error(f'Error when calling remote API: {str(e)}')
            raise e

    def _collect_stream_response(self, response_stream: List[ChatCompletionChunk]) -> ChatCompletion:
        collected_chunks = []
        collected_messages = defaultdict(list)
        collected_reasoning = defaultdict(list)
        collected_tool_calls = defaultdict(dict)

        for chunk in response_stream:
            collected_chunks.append(chunk)
            for choice in chunk.choices:
                # Handle reasoning content
                if hasattr(choice.delta, 'reasoning_content') and choice.delta.reasoning_content is not None:
                    collected_reasoning[choice.index].append(choice.delta.reasoning_content)

                # Handle regular content
                if choice.delta.content is not None:
                    collected_messages[choice.index].append(choice.delta.content)

                # Handle tool calls
                if hasattr(choice.delta, 'tool_calls') and choice.delta.tool_calls:
                    for tool_call in choice.delta.tool_calls:
                        tool_id = tool_call.index

                        # Initialize tool call if not present
                        if tool_id not in collected_tool_calls[choice.index]:
                            collected_tool_calls[choice.index][tool_id] = {
                                'id': tool_call.id if hasattr(tool_call, 'id') and tool_call.id else None,
                                'type': tool_call.type if hasattr(tool_call, 'type') and tool_call.type else None,
                                'function': {
                                    'name': '',
                                    'arguments': ''
                                }
                            }

                        # Update tool call with new chunks
                        if hasattr(tool_call, 'function'):
                            if hasattr(tool_call.function, 'name') and tool_call.function.name:
                                collected_tool_calls[
                                    choice.index][tool_id]['function']['name'] = tool_call.function.name

                            if hasattr(tool_call.function, 'arguments') and tool_call.function.arguments:
                                collected_tool_calls[
                                    choice.index][tool_id]['function']['arguments'] += tool_call.function.arguments

                        # Update ID if it was received later
                        if hasattr(tool_call, 'id') and tool_call.id:
                            collected_tool_calls[choice.index][tool_id]['id'] = tool_call.id

        # Get all unique choice indices from all collections
        all_indices = set(collected_messages.keys()) | set(collected_reasoning.keys()) | set(
            collected_tool_calls.keys())

        choices = []
        for index in all_indices:
            full_reply_content = ''.join(collected_messages.get(index, []))
            reasoning = ''.join(collected_reasoning.get(index, []))

            # Process tool_calls for this choice if any exists
            tool_calls_list = None
            if index in collected_tool_calls and collected_tool_calls[index]:
                tool_calls_list = list(collected_tool_calls[index].values())
                # Filter out any tool calls with None id (incomplete tool calls)
                tool_calls_list = [tc for tc in tool_calls_list if tc['id'] is not None]

            # use the finish_reason from the last chunk that generated this choice
            finish_reason = None
            for chunk in reversed(collected_chunks):
                if chunk.choices and chunk.choices[0].index == index:
                    finish_reason = chunk.choices[0].finish_reason
                    break

            message_kwargs = {'role': 'assistant', 'content': full_reply_content}

            if reasoning:
                message_kwargs['reasoning_content'] = reasoning

            if tool_calls_list:
                message_kwargs['tool_calls'] = tool_calls_list

            choice = Choice(
                finish_reason=finish_reason or 'stop', index=index, message=ChatCompletionMessage(**message_kwargs))
            choices.append(choice)

        # build the final completion object
        return ChatCompletion(
            id=collected_chunks[0].id,
            choices=choices,
            created=collected_chunks[0].created,
            model=collected_chunks[0].model,
            object='chat.completion',
            usage=collected_chunks[-1].usage  # use the usage from the last chunk
        )
