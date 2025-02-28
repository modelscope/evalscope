import openai
from collections import defaultdict
from openai.types.chat import ChatCompletion, ChatCompletionChunk
from openai.types.chat.chat_completion import ChatCompletionMessage, Choice
from typing import List, Optional, Union

from evalscope.models.base_adapter import BaseModelAdapter
from evalscope.models.register import register_model_adapter
from evalscope.utils.logger import get_logger

logger = get_logger()


@register_model_adapter('server')
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

        self.seed = kwargs.get('seed', None)
        self.timeout = kwargs.get('timeout', 60)
        self.stream = kwargs.get('stream', False)
        self.model_cfg = {'api_url': api_url, 'model_id': model_id, 'api_key': api_key}
        super().__init__(model=None, model_cfg=self.model_cfg, **kwargs)

    def predict(self, inputs: List[Union[str, dict, list]], infer_cfg: dict = None) -> List[dict]:
        """
        Model prediction func.

        Args:
            inputs (List[Union[str, dict, list]]): The input data.
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
        data: list = input_item['data']
        if isinstance(data[0], tuple):  # for truthful_qa and hellaswag
            query = '\n'.join(''.join(item) for item in data)
            system_prompt = input_item.get('system_prompt', None)
        else:
            query = data[0]
            system_prompt = input_item.get('system_prompt', None)

        content = self.make_request_content(query, system_prompt)
        request_json = self.make_request(content, infer_cfg)
        response = self.send_request(request_json)
        return response

    def make_request_content(self, query: str, system_prompt: Optional[str] = None) -> list:
        """
        Make request content for OpenAI API.
        """
        messages = []
        if system_prompt:
            messages.append({'role': 'system', 'content': system_prompt})

        messages.append({'role': 'user', 'content': query})

        return messages

    def make_request(self, content: list, infer_cfg: dict = {}) -> dict:
        """Make request to remote API."""
        # Format request JSON according to OpenAI API format
        from evalscope.config import DEFAULT_GENERATION_CONFIG
        if infer_cfg == DEFAULT_GENERATION_CONFIG:
            infer_cfg = {
                'max_tokens': 2048,
                'temperature': 0.0,
            }

        request_json = {'model': self.model_id, 'messages': content, **infer_cfg}

        if self.timeout:
            request_json['timeout'] = self.timeout

        if self.stream:
            request_json['stream'] = self.stream
            request_json['stream_options'] = {'include_usage': True}

        logger.debug(f'Request to remote API: {request_json}')
        return request_json

    def send_request(self, request_json: dict) -> dict:
        try:
            response = self.client.chat.completions.create(**request_json)

            if self.stream:
                response = self._collect_stream_response(response)

            return response.model_dump(exclude_unset=True)
        except Exception as e:
            logger.error(f'Error when calling remote API: {str(e)}')
            raise

    def _collect_stream_response(self, response_stream: List[ChatCompletionChunk]) -> ChatCompletion:
        collected_chunks = []
        collected_messages = defaultdict(list)

        for chunk in response_stream:
            collected_chunks.append(chunk)
            for choice in chunk.choices:
                if choice.delta.content is not None:
                    collected_messages[choice.index].append(choice.delta.content)

        choices = []
        for index, messages in collected_messages.items():
            full_reply_content = ''.join(messages)

            # use the finish_reason from the last chunk that generated this choice
            finish_reason = 'stop'
            for chunk in reversed(collected_chunks):
                if chunk.choices and chunk.choices[0].index == index:
                    finish_reason = chunk.choices[0].finish_reason
                    break

            choice = Choice(
                finish_reason=finish_reason,
                index=index,
                message=ChatCompletionMessage(role='assistant', content=full_reply_content))
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
