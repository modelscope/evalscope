import json
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import aiohttp

from evalscope.perf.arguments import Arguments
from evalscope.perf.multi_turn_args import _sample_int_or_range
from evalscope.perf.plugin.api.base import ApiPluginBase
from evalscope.perf.plugin.api.default_api import StreamedResponseHandler
from evalscope.perf.plugin.registry import register_api
from evalscope.perf.utils.benchmark_util import BenchmarkData, is_stream_body
from evalscope.utils.io_utils import json_to_dict


@register_api('anthropic')
class AnthropicMessagesPlugin(ApiPluginBase):
    """Anthropic Messages API plugin using the standard perf HTTP pipeline."""

    def build_request(self, messages: Union[List[Dict], str, Dict], param: Arguments = None) -> Dict:
        """Convert dataset input or fill unspecified fields in a native request body."""
        param = param or self.param
        template = param.query_template
        if isinstance(messages, dict):
            if template is not None:
                raise ValueError(
                    'A complete Anthropic request body must be passed directly, not as query-template input.'
                )
            return self._compose_request(messages, param, preserve_existing=True)

        from evalscope.api.messages import dict_to_chat_message
        from evalscope.models.utils.anthropic import anthropic_chat_messages, merge_anthropic_system_messages

        query = {}
        if template is not None:
            query = json_to_dict(template[1:]) if template.startswith('@') else json.loads(template)
        raw_messages = [{'role': 'user', 'content': messages}] if isinstance(messages, str) else messages
        system, query['messages'] = anthropic_chat_messages([dict_to_chat_message(message) for message in raw_messages])
        if system:
            query['system'] = merge_anthropic_system_messages(query.get('system'), system)
        return self._compose_request(query, param)

    def parse_responses(self, responses: List[Dict], request: str = None, **kwargs) -> Tuple[int, int]:
        """Extract prompt and completion token counts from a Messages response."""
        prompt_tokens, completion_tokens, _ = self._parse_usage(responses)
        return prompt_tokens, completion_tokens

    async def process_request(
        self, client_session: aiohttp.ClientSession, url: str, headers: Dict, body: Dict
    ) -> BenchmarkData:
        """Read a Messages response and validate it before marking the request successful."""
        headers = {'Content-Type': 'application/json', **headers}
        data = json.dumps(body, ensure_ascii=False)
        output = BenchmarkData(start_time=time.perf_counter(), request=data)
        try:
            async with client_session.post(url=url, data=data, headers=headers) as response:
                if response.status != 200:
                    output.status_code = response.status
                    output.error = await response.text()
                    output.is_stream = is_stream_body(body)
                    return output

                if 'text/event-stream' in response.headers.get('Content-Type', ''):
                    await self._process_stream(response, output)
                else:
                    await self._process_message(response, output)
                output.prompt_tokens, output.completion_tokens, output.real_cached_tokens = self._parse_usage(
                    output.response_messages
                )
                output.success = True
        except (ValueError, aiohttp.ContentTypeError) as exc:
            output.error = f'Invalid Anthropic Messages response: {exc}'
            output.is_stream = output.is_stream or is_stream_body(body)
        return output

    @staticmethod
    async def _process_message(response: aiohttp.ClientResponse, output: BenchmarkData) -> None:
        payload = await response.json()
        output.completed_time = time.perf_counter()
        output.query_latency = output.completed_time - output.start_time
        output.first_chunk_latency = output.query_latency
        if not isinstance(payload, dict):
            raise ValueError('Expected a JSON message object.')
        if payload.get('type') == 'error' or payload.get('error') is not None:
            raise ValueError(json.dumps(payload, ensure_ascii=False))
        content = payload.get('content')
        if payload.get('type') != 'message' or not isinstance(content, list):
            raise ValueError('Expected a Messages API response with a content list.')
        output.generated_text = ''.join(_block_text(block) for block in content)
        output.response_messages.append(payload)

    async def _process_stream(self, response: aiohttp.ClientResponse, output: BenchmarkData) -> None:
        output.is_stream = True
        handler = StreamedResponseHandler()
        text_parts = []
        last_event_time = output.start_time
        last_output_timestamp: Optional[float] = None
        stop_reason: Optional[str] = None
        message_started = False
        message_stopped = False
        async for chunk_bytes in response.content.iter_any():
            for message in handler.add_chunk(chunk_bytes):
                payload = json.loads(message.removeprefix('data:'))
                if not isinstance(payload, dict):
                    raise ValueError('Expected a JSON object in the Anthropic SSE event.')

                timestamp = time.perf_counter()
                event_type = payload.get('type')
                if event_type == 'message_start':
                    message_started = True
                elif event_type in ('content_block_start', 'content_block_delta'):
                    text, has_output = self._stream_output(payload)
                    text_parts.append(text)
                    if has_output:
                        if last_output_timestamp is None:
                            output.first_chunk_latency = timestamp - output.start_time
                        else:
                            output.inter_chunk_latency.append(timestamp - last_output_timestamp)
                        last_output_timestamp = timestamp
                elif event_type == 'message_delta':
                    delta = payload.get('delta')
                    if not isinstance(delta, dict):
                        raise ValueError('Expected a delta object in the Anthropic message_delta event.')
                    stop_reason = delta.get('stop_reason') or stop_reason
                elif event_type == 'message_stop':
                    message_stopped = True
                elif event_type == 'error':
                    raise ValueError(json.dumps(payload, ensure_ascii=False))

                if event_type != 'ping':
                    output.response_messages.append(payload)
                last_event_time = timestamp

        if any(line.startswith('data:') for line in handler.buffer.split('\n')):
            raise ValueError('Anthropic SSE stream ended with incomplete data.')
        if not message_started:
            raise ValueError('SSE stream ended without an Anthropic message_start event.')
        if not message_stopped:
            raise ValueError('SSE stream ended without an Anthropic message_stop event.')
        if stop_reason is None:
            raise ValueError('Anthropic stream ended without a stop reason.')
        output.generated_text = ''.join(text_parts)
        output.completed_time = last_event_time
        output.query_latency = last_event_time - output.start_time

    @staticmethod
    def _compose_request(payload: Dict, param: Arguments, preserve_existing: bool = False) -> Dict:
        options = {
            'model': param.model,
            'stream': param.stream,
            'temperature': param.temperature,
            'top_p': param.top_p,
            'top_k': param.top_k,
            'stop_sequences': param.stop,
        }
        if param.max_tokens is not None:
            options['max_tokens'] = _sample_int_or_range(param.max_tokens)
        overrides = {key: value for key, value in options.items() if value is not None}
        if preserve_existing:
            return {**(param.extra_args or {}), 'max_tokens': 4096, **overrides, **payload}
        return {'max_tokens': 4096, **payload, **overrides, **(param.extra_args or {})}

    @staticmethod
    def _parse_usage(responses: List[Dict]) -> Tuple[int, int, Optional[int]]:
        usage = {}
        for event in responses:
            source = event.get('message') if event.get('type') == 'message_start' else event
            raw = source.get('usage') if isinstance(source, dict) else None
            if isinstance(raw, dict):
                usage.update({key: value for key, value in raw.items() if value is not None})
        input_tokens = usage.get('input_tokens')
        output_tokens = usage.get('output_tokens')
        cache_read = usage.get('cache_read_input_tokens')
        cache_creation = usage.get('cache_creation_input_tokens', 0)
        counts = (input_tokens, output_tokens, 0 if cache_read is None else cache_read, cache_creation)
        if any(type(count) is not int or count < 0 for count in counts):
            raise ValueError('Missing or invalid Anthropic token usage.')
        return input_tokens + (cache_read or 0) + cache_creation, output_tokens, cache_read

    @staticmethod
    def _stream_output(payload: Dict) -> Tuple[str, bool]:
        is_start = payload['type'] == 'content_block_start'
        block = payload.get('content_block' if is_start else 'delta')
        text = _block_text(block)
        if is_start:
            has_output = block.get('type') == 'redacted_thinking' or (
                block.get('type') in ('tool_use', 'server_tool_use') and bool(block.get('name'))
            )
        else:
            has_output = bool(block.get('partial_json'))
        return text, bool(text or has_output)


def _block_text(block: Any) -> str:
    if not isinstance(block, dict):
        raise ValueError('Expected an Anthropic content block object.')
    text = block.get('text') or block.get('thinking') or ''
    if not isinstance(text, str):
        raise ValueError('Expected a string in the Anthropic content block.')
    return text
