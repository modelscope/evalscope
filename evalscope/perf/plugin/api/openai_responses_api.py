import json
import time
from collections import defaultdict
from typing import Any, Dict, List, Tuple, Union

from evalscope.models.utils.openai_responses import (
    normalize_responses_input,
    response_text_from_dict,
    response_usage_from_dict,
)
from evalscope.perf.arguments import Arguments
from evalscope.perf.multi_turn_args import _sample_int_or_range
from evalscope.perf.plugin.api.default_api import DefaultApiPlugin, StreamedResponseHandler
from evalscope.perf.plugin.datasets.utils import load_tokenizer
from evalscope.perf.plugin.registry import register_api
from evalscope.perf.utils.benchmark_util import BenchmarkData
from evalscope.utils.logger import get_logger

logger = get_logger()


@register_api(['openai_responses', 'openai_response', 'responses'])
class OpenAIResponsesPlugin(DefaultApiPlugin):
    """OpenAI official Responses API plugin."""

    def __init__(self, param: Arguments):
        super().__init__(param=param)
        if param.tokenizer_path is not None:
            self.tokenizer = load_tokenizer(param.tokenizer_path)
        else:
            self.tokenizer = None

    def build_request(self, messages: Union[List[Dict], str, Dict], param: Arguments = None) -> Dict:
        param = param or self.param
        try:
            if param.query_template is not None:
                query = self._load_query_template(param.query_template)
                query['input'] = normalize_responses_input(messages)
            elif isinstance(messages, dict):
                query = dict(messages)
                if 'messages' in query and 'input' not in query:
                    query['input'] = normalize_responses_input(query.pop('messages'))
            else:
                query = {'input': normalize_responses_input(messages)}
            return self._compose_query_from_parameter(query, param)
        except Exception as e:
            logger.exception(e)
            return None

    def parse_responses(self, responses: List[Dict], request: str = None, **kwargs) -> Tuple[int, int]:
        if not responses:
            logger.error('Received empty response list (responses=[]) from OpenAI Responses API.')
            return 0, 0

        for response in reversed(responses):
            payload = response.get('response', response)
            usage = response_usage_from_dict(payload)
            if usage is not None:
                return usage

        if self.tokenizer is None:
            raise ValueError(
                'Error: Unable to retrieve usage information from OpenAI Responses API response and no tokenizer was '
                'specified. Please ensure the API returns usage or set --tokenizer-path.'
            )

        input_tokens = self._count_input_tokens(request)
        output_tokens = 0
        for idx, choice_contents in self._collect_output_text(responses).items():
            output_tokens += len(self.tokenizer.encode(''.join(choice_contents), add_special_tokens=False))
        return input_tokens, output_tokens

    async def process_request(self, client_session, url: str, headers: Dict, body: Dict) -> BenchmarkData:
        headers = {'Content-Type': 'application/json', **headers}
        data = json.dumps(body, ensure_ascii=False)

        output = BenchmarkData()
        ttft = 0.0
        generated_text = ''
        st = time.perf_counter()
        output.start_time = st
        output.request = data
        most_recent_timestamp = st
        try:
            async with client_session.post(url=url, data=data, headers=headers) as response:
                content_type = response.headers.get('Content-Type', '')
                if response.status != 200:
                    output.status_code = response.status
                    try:
                        err_payload = await response.json()
                        output.error = json.dumps(err_payload, ensure_ascii=False)
                    except Exception:
                        output.error = await response.text()
                    output.success = False
                    return output

                if 'text/event-stream' in content_type:
                    handler = StreamedResponseHandler()
                    stream_failed = False
                    async for chunk_bytes in response.content.iter_any():
                        if not chunk_bytes:
                            continue
                        messages = handler.add_chunk(chunk_bytes)
                        for message in messages:
                            if message.startswith(':'):
                                continue
                            chunk = _extract_sse_data(message)
                            if not chunk:
                                continue
                            if chunk == '[DONE]':
                                continue

                            timestamp = time.perf_counter()
                            payload = json.loads(chunk)
                            event_type = payload.get('type')
                            delta = payload.get('delta') or ''
                            if event_type in _DELTA_EVENT_TYPES and delta:
                                if ttft == 0.0:
                                    ttft = timestamp - st
                                    output.first_chunk_latency = ttft
                                else:
                                    output.inter_chunk_latency.append(timestamp - most_recent_timestamp)
                                generated_text += delta
                            elif event_type == 'response.completed':
                                response_payload = payload.get('response', {})
                                usage = response_usage_from_dict(response_payload)
                                if usage is not None:
                                    output.prompt_tokens, output.completion_tokens = usage
                                    self._set_cached_tokens(output, response_payload.get('usage', {}))
                                if not generated_text:
                                    generated_text = response_text_from_dict(response_payload)
                            elif event_type == 'response.failed':
                                stream_failed = True
                                output.error = json.dumps(payload.get('response', payload), ensure_ascii=False)

                            output.response_messages.append(payload)
                            most_recent_timestamp = timestamp

                    output.generated_text = generated_text
                    output.success = not stream_failed
                    output.completed_time = most_recent_timestamp
                    output.query_latency = most_recent_timestamp - st
                    return output

                payload: Any
                try:
                    payload = await response.json()
                except Exception:
                    payload = await response.text()

                timestamp = time.perf_counter()
                output.completed_time = timestamp
                output.query_latency = timestamp - st
                output.first_chunk_latency = output.query_latency

                if isinstance(payload, dict):
                    output.generated_text = response_text_from_dict(payload)
                    usage = response_usage_from_dict(payload)
                    if usage is not None:
                        output.prompt_tokens, output.completion_tokens = usage
                        self._set_cached_tokens(output, payload.get('usage', {}))
                    output.response_messages.append(payload)
                else:
                    output.generated_text = str(payload)
                    output.response_messages.append(payload)
                output.success = True
                return output
        except Exception:
            import sys
            import traceback

            output.success = False
            output.error = ''.join(traceback.format_exception(*sys.exc_info()))
            logger.error(output.error)
            return output

    @staticmethod
    def _load_query_template(query_template: str) -> Dict:
        if query_template.startswith('@'):
            file_path = query_template[1:]
            with open(file_path, 'r') as file:
                return json.load(file)
        return json.loads(query_template)

    @staticmethod
    def _set_cached_tokens(output: Any, usage: Dict[str, Any]) -> None:
        details = usage.get('input_tokens_details') or usage.get('prompt_tokens_details')
        if details and isinstance(details, dict):
            cached = details.get('cached_tokens')
            if cached is not None:
                output.real_cached_tokens = cached

    def _compose_query_from_parameter(self, payload: Dict, param: Arguments) -> Dict:
        payload['model'] = param.model
        if param.max_tokens is not None:
            payload['max_output_tokens'] = _sample_int_or_range(param.max_tokens)
        if param.stream is not None:
            payload['stream'] = param.stream
        if param.temperature is not None:
            payload['temperature'] = param.temperature
        if param.top_p is not None:
            payload['top_p'] = param.top_p
        if param.n_choices is not None and param.n_choices > 1:
            logger.warning('OpenAI Responses API does not support n_choices > 1; ignoring --n-choices.')
        if param.extra_args is not None:
            payload.update(param.extra_args)
        return payload

    def _count_input_tokens(self, request_str: str) -> int:
        request = json.loads(request_str)
        input_value = request.get('input', '')
        return self._count_input_value(input_value)

    def _count_input_value(self, value: Any) -> int:
        if isinstance(value, str):
            return len(self.tokenizer.encode(value, add_special_tokens=False))
        if isinstance(value, list):
            return sum(self._count_input_value(item) for item in value)
        if not isinstance(value, dict):
            return 0
        if isinstance(value.get('content'), str):
            return self._count_input_value(value['content'])
        if isinstance(value.get('content'), list):
            return self._count_input_value(value['content'])
        if value.get('type') in ('input_text', 'output_text'):
            return len(self.tokenizer.encode(value.get('text', ''), add_special_tokens=False))
        return 0

    @staticmethod
    def _collect_output_text(responses: List[Dict]) -> Dict[int, List[str]]:
        contents = defaultdict(list)
        has_delta = any(response.get('type') in _DELTA_EVENT_TYPES for response in responses)
        for response in responses:
            if response.get('type') in _DELTA_EVENT_TYPES:
                contents[0].append(response.get('delta') or '')
            elif not has_delta:
                text = response_text_from_dict(response.get('response', response))
                if text:
                    contents[0].append(text)
        return contents


_DELTA_EVENT_TYPES = {
    'response.output_text.delta',
    'response.refusal.delta',
    'response.reasoning_text.delta',
    'response.reasoning_summary_text.delta',
}


def _extract_sse_data(message: str) -> str:
    data_lines = []
    for line in message.splitlines():
        line = line.strip()
        if not line or line.startswith(':'):
            continue
        if line.startswith('data:'):
            data_lines.append(line.removeprefix('data:').strip())
    if data_lines:
        return '\n'.join(data_lines).strip()
    return message.removeprefix('data:').strip()
