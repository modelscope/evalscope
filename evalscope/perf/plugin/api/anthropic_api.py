import json
import sys
import time
import traceback
from typing import Any, Dict, List, Optional, Tuple, Union

import aiohttp

from evalscope.perf.arguments import Arguments
from evalscope.perf.multi_turn_args import _sample_int_or_range
from evalscope.perf.plugin.api.default_api import DefaultApiPlugin, StreamedResponseHandler
from evalscope.perf.plugin.datasets.utils import load_tokenizer
from evalscope.perf.plugin.registry import register_api
from evalscope.perf.utils.benchmark_util import BenchmarkData, is_stream_body
from evalscope.utils.import_utils import check_import
from evalscope.utils.logger import get_logger

logger = get_logger()


def _prompt_tokens_from_usage(usage: Dict) -> Optional[int]:
    fields = ('input_tokens', 'cache_read_input_tokens', 'cache_creation_input_tokens')
    if all(usage.get(field) is None for field in fields):
        return None
    return sum(usage.get(field) or 0 for field in fields)


@register_api('anthropic')
class AnthropicMessagesPlugin(DefaultApiPlugin):
    """Anthropic Messages API plugin using the standard perf HTTP pipeline."""

    def __init__(self, param: Arguments):
        check_import('anthropic', package='anthropic', raise_error=True, feature_name='Anthropic performance testing')
        super().__init__(param=param)
        self.tokenizer = load_tokenizer(param.tokenizer_path) if param.tokenizer_path is not None else None

    def build_request(self, messages: Union[List[Dict], str, Dict], param: Arguments = None) -> Dict:
        param = param or self.param
        if param.tokenize_prompt:
            raise ValueError('Anthropic Messages API does not support --tokenize-prompt.')
        if param.query_template is not None:
            query = self._load_query_template(param.query_template)
            self._set_input_messages(query, messages)
        elif isinstance(messages, dict):
            query = dict(messages)
        else:
            query = {}
            self._set_input_messages(query, messages)
        preserve_existing = param.query_template is None and isinstance(messages, dict)
        return self._compose_query_from_parameter(query, param, preserve_existing=preserve_existing)

    def parse_responses(self, responses: List[Dict], request: str = None, **kwargs) -> Tuple[int, int]:
        if not responses:
            logger.error('Received empty response list (responses=[]) from Anthropic Messages API.')
            return 0, 0

        merged_usage: Dict = {}
        for response in responses:
            usage_source = response.get('message', {}) if response.get('type') == 'message_start' else response
            usage = usage_source.get('usage') if isinstance(usage_source, dict) else None
            if not isinstance(usage, dict):
                continue
            for field in ('input_tokens', 'cache_read_input_tokens', 'cache_creation_input_tokens', 'output_tokens'):
                if usage.get(field) is not None:
                    merged_usage[field] = usage[field]
        prompt_tokens = _prompt_tokens_from_usage(merged_usage)
        output_tokens = merged_usage.get('output_tokens')
        if prompt_tokens is not None or output_tokens is not None:
            return prompt_tokens or 0, output_tokens or 0

        if self.tokenizer is None:
            raise ValueError(
                'Unable to retrieve usage from the Anthropic Messages API response and no tokenizer was specified. '
                'Please ensure the API returns usage or set --tokenizer-path.'
            )
        request_payload = json.loads(request or '{}')
        input_count = self._count_input_value(request_payload.get('system'))
        input_count += self._count_input_value(request_payload.get('messages'))
        output_count = sum(
            len(self.tokenizer.encode(self._response_text(response), add_special_tokens=False))
            for response in responses
        )
        return input_count, output_count

    async def process_request(
        self, client_session: aiohttp.ClientSession, url: str, headers: Dict, body: Dict
    ) -> BenchmarkData:
        headers = {'Content-Type': 'application/json', **headers}
        data = json.dumps(body, ensure_ascii=False)
        output = BenchmarkData()
        st = time.perf_counter()
        output.start_time = st
        output.request = data
        try:
            async with client_session.post(url=url, data=data, headers=headers) as response:
                content_type = response.headers.get('Content-Type', '')
                if response.status != 200:
                    output.status_code = response.status
                    try:
                        error_payload = await response.json()
                        output.error = json.dumps(error_payload, ensure_ascii=False)
                    except Exception:
                        output.error = await response.text()
                    output.is_stream = is_stream_body(body)
                    return output

                if 'text/event-stream' in content_type:
                    return await self._process_stream(response, output, st)

                timestamp = time.perf_counter()
                output.completed_time = timestamp
                output.query_latency = timestamp - st
                output.first_chunk_latency = output.query_latency
                try:
                    payload: Any = await response.json()
                except Exception:
                    payload = await response.text()
                if isinstance(payload, dict):
                    if payload.get('type') == 'error' or payload.get('error') is not None:
                        output.error = json.dumps(payload, ensure_ascii=False)
                    output.generated_text = self._response_text(payload)
                    self._apply_usage(output, payload.get('usage'))
                    output.response_messages.append(payload)
                else:
                    output.generated_text = str(payload)
                    output.response_messages.append(payload)
                output.success = output.error is None
                return output
        except Exception:
            output.success = False
            output.error = ''.join(traceback.format_exception(*sys.exc_info()))
            output.is_stream = output.is_stream or is_stream_body(body)
            logger.error(output.error)
            return output

    async def _process_stream(
        self, response: aiohttp.ClientResponse, output: BenchmarkData, st: float
    ) -> BenchmarkData:
        output.is_stream = True
        handler = StreamedResponseHandler()
        generated_text = ''
        most_recent_timestamp = st
        last_output_timestamp: Optional[float] = None
        stop_reason: Optional[str] = None
        message_started = False
        message_stopped = False
        stream_usage: Dict = {}
        async for chunk_bytes in response.content.iter_any():
            if not chunk_bytes:
                continue
            for message in handler.add_chunk(chunk_bytes):
                try:
                    payload = json.loads(message.removeprefix('data:').strip())
                except json.JSONDecodeError:
                    output.error = 'Invalid JSON in Anthropic SSE event.'
                    continue
                if not isinstance(payload, dict):
                    output.error = 'Invalid non-object payload in Anthropic SSE event.'
                    continue

                timestamp = time.perf_counter()
                event_type = payload.get('type')
                if event_type == 'message_start':
                    message_started = True
                    self._apply_usage(output, payload.get('message', {}).get('usage'), stream_usage)
                elif event_type in ('content_block_start', 'content_block_delta'):
                    text, has_output = self._stream_output(payload)
                    if has_output:
                        if last_output_timestamp is None:
                            output.first_chunk_latency = timestamp - st
                        else:
                            output.inter_chunk_latency.append(timestamp - last_output_timestamp)
                        last_output_timestamp = timestamp
                        generated_text += text
                elif event_type == 'message_delta':
                    stop_reason = payload.get('delta', {}).get('stop_reason') or stop_reason
                    self._apply_usage(output, payload.get('usage'), stream_usage)
                elif event_type == 'message_stop':
                    message_stopped = True
                elif event_type == 'error':
                    output.error = json.dumps(payload, ensure_ascii=False)

                if event_type != 'ping':
                    output.response_messages.append(payload)
                most_recent_timestamp = timestamp

        has_incomplete_data = any(line.startswith('data:') for line in handler.buffer.split('\n'))
        if has_incomplete_data and output.error is None:
            output.error = 'Anthropic SSE stream ended with malformed or incomplete data.'
        if not message_started and output.error is None:
            output.error = 'SSE stream ended without an Anthropic message_start event.'
        elif stop_reason is None and output.error is None:
            output.error = 'Anthropic stream ended without a stop reason.'
        elif not message_stopped and output.error is None:
            output.error = 'SSE stream ended without an Anthropic message_stop event.'
        output.generated_text = generated_text
        output.success = output.error is None
        output.completed_time = most_recent_timestamp
        output.query_latency = most_recent_timestamp - st
        return output

    @staticmethod
    def _load_query_template(query_template: str) -> Dict:
        if query_template.startswith('@'):
            with open(query_template[1:], 'r', encoding='utf-8') as file:
                return json.load(file)
        return json.loads(query_template)

    @staticmethod
    def _convert_messages(messages: Union[List[Dict], str]) -> Tuple[Any, List[Dict]]:
        from evalscope.api.messages import dict_to_chat_message
        from evalscope.models.utils.anthropic import anthropic_chat_messages

        raw_messages = [{'role': 'user', 'content': messages}] if isinstance(messages, str) else messages
        native_messages = [dict_to_chat_message(message) for message in raw_messages]
        return anthropic_chat_messages(native_messages)

    def _set_input_messages(self, query: Dict, messages: Union[List[Dict], str, Dict]) -> None:
        if isinstance(messages, dict):
            raise ValueError('A complete Anthropic request body must be passed directly, not as query-template input.')
        system, converted = self._convert_messages(messages)
        query['messages'] = converted
        if system:
            query['system'] = self._merge_system(query.get('system'), system)

    @staticmethod
    def _merge_system(current: Any, new: Any) -> Any:
        if current is None:
            return new
        current_blocks = [{'type': 'text', 'text': current}] if isinstance(current, str) else current
        new_blocks = [{'type': 'text', 'text': new}] if isinstance(new, str) else new
        return [*current_blocks, {'type': 'text', 'text': '\n\n'}, *new_blocks]

    @staticmethod
    def _compose_query_from_parameter(payload: Dict, param: Arguments, preserve_existing: bool = False) -> Dict:
        def _set(key: str, value: Any) -> None:
            if preserve_existing:
                payload.setdefault(key, value)
            else:
                payload[key] = value

        if param.model is not None:
            _set('model', param.model)
        if param.max_tokens is not None:
            _set('max_tokens', _sample_int_or_range(param.max_tokens))
        else:
            payload.setdefault('max_tokens', 4096)
        if param.stream is not None:
            _set('stream', param.stream)
        if param.temperature is not None:
            _set('temperature', param.temperature)
        if param.top_p is not None:
            _set('top_p', param.top_p)
        if param.top_k is not None:
            _set('top_k', param.top_k)
        if param.stop is not None:
            _set('stop_sequences', [param.stop] if isinstance(param.stop, str) else param.stop)
        if param.n_choices is not None and param.n_choices > 1:
            logger.warning('Anthropic Messages API does not support n_choices > 1; ignoring --n-choices.')
        if param.extra_args is not None:
            for key, value in param.extra_args.items():
                if preserve_existing:
                    payload.setdefault(key, value)
                else:
                    payload[key] = value
        return payload

    @staticmethod
    def _apply_usage(output: BenchmarkData, usage: Any, accumulated: Optional[Dict] = None) -> None:
        if not isinstance(usage, dict):
            return
        if accumulated is not None:
            for field in ('input_tokens', 'cache_read_input_tokens', 'cache_creation_input_tokens', 'output_tokens'):
                if usage.get(field) is not None:
                    accumulated[field] = usage[field]
            usage = accumulated
        prompt_tokens = _prompt_tokens_from_usage(usage)
        if prompt_tokens is not None:
            output.prompt_tokens = prompt_tokens
        if usage.get('output_tokens') is not None:
            output.completion_tokens = usage['output_tokens']
        cached = usage.get('cache_read_input_tokens')
        details = usage.get('prompt_tokens_details')
        if cached is None and isinstance(details, dict):
            cached = details.get('cached_tokens')
        if cached is not None:
            output.real_cached_tokens = cached

    @staticmethod
    def _stream_output(payload: Dict) -> Tuple[str, bool]:
        event_type = payload.get('type')
        block = payload.get('content_block', {}) if event_type == 'content_block_start' else payload.get('delta', {})
        text = block.get('text') or block.get('thinking') or ''
        has_tool_output = event_type == 'content_block_delta' and bool(block.get('partial_json'))
        has_redacted_thinking = event_type == 'content_block_start' and block.get('type') == 'redacted_thinking'
        return text, bool(text or has_tool_output or has_redacted_thinking)

    @classmethod
    def _response_text(cls, payload: Dict) -> str:
        if payload.get('type') == 'message_start':
            return cls._response_text(payload.get('message', {}))
        content = payload.get('content')
        if isinstance(content, list):
            return ''.join(block.get('text') or block.get('thinking') or '' for block in content)
        delta = payload.get('delta')
        if isinstance(delta, dict):
            return delta.get('text') or delta.get('thinking') or ''
        return ''

    def _count_input_value(self, value: Any) -> int:
        if value is None:
            return 0
        if isinstance(value, str):
            return len(self.tokenizer.encode(value, add_special_tokens=False))
        if isinstance(value, list):
            return sum(self._count_input_value(item) for item in value)
        if not isinstance(value, dict):
            return 0
        if 'messages' in value or 'system' in value:
            return self._count_input_value(value.get('messages')) + self._count_input_value(value.get('system'))
        if 'content' in value:
            return self._count_input_value(value['content'])
        if value.get('type') in ('text', 'input_text', 'output_text'):
            return self._count_input_value(value.get('text', ''))
        if value.get('type') in ('thinking', 'thinking_delta'):
            return self._count_input_value(value.get('thinking', ''))
        return 0
