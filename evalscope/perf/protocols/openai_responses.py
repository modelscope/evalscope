from __future__ import annotations

import json
from typing import Any, Dict, Optional

from evalscope.perf.config.models import GenerationConfig
from evalscope.perf.domain.observation import TokenSource, TransportEvent
from evalscope.perf.domain.workload import SingleTurnItem
from evalscope.perf.protocols.base import ProtocolAdapter, ProtocolMeta, ProtocolState
from evalscope.perf.transport.base import HttpRequest


class OpenAIResponsesProtocol(ProtocolAdapter):
    meta = ProtocolMeta(name='openai_responses', endpoint='responses')

    def build_request(
        self,
        item: SingleTurnItem,
        generation: GenerationConfig,
        max_tokens: Optional[int] = None,
    ) -> HttpRequest:
        payload: Dict[str, Any] = {'model': self.target.model, 'input': item.messages}
        self._apply_generation(payload, generation, max_tokens)
        payload['max_output_tokens'] = payload.pop('max_tokens')
        headers = dict(self.target.headers)
        if self.target.api_key:
            headers['Authorization'] = f'Bearer {self.target.api_key}'
        return HttpRequest(url=self.url(), body=payload, headers=headers)

    @staticmethod
    def _usage(state: ProtocolState, payload: Dict[str, Any]) -> None:
        usage = payload.get('usage') or {}
        if not usage:
            return
        state.prompt_tokens = usage.get('input_tokens')
        state.completion_tokens = usage.get('output_tokens')
        state.cached_tokens = (usage.get('input_tokens_details') or {}).get('cached_tokens')
        if state.prompt_tokens is not None:
            state.prompt_token_source = TokenSource.SERVER_REPORTED
        if state.completion_tokens is not None:
            state.completion_token_source = TokenSource.SERVER_REPORTED
        if state.cached_tokens is not None:
            state.cache_token_source = TokenSource.SERVER_REPORTED

    def _consume_payload(self, state: ProtocolState, payload: Any, timestamp: float) -> None:
        state.response_payloads.append(payload)
        if not isinstance(payload, dict):
            state.generated_text += str(payload)
            return
        event_type = payload.get('type')
        delta = payload.get('delta') or ''
        if event_type in {'response.output_text.delta', 'response.reasoning_text.delta'} and delta:
            if state.first_token_time is None:
                state.first_token_time = timestamp
            state.chunk_times.append(timestamp)
            state.generated_text += delta
        response = payload.get('response', payload)
        self._usage(state, response)
        if not state.generated_text and isinstance(response, dict):
            for output in response.get('output') or []:
                for content in output.get('content') or []:
                    state.generated_text += content.get('text') or ''

    def consume_event(self, state: ProtocolState, event: TransportEvent) -> None:
        if self._consume_common(state, event):
            return
        if event.kind == 'json':
            self._consume_payload(state, event.data, event.timestamp)
        elif event.kind == 'sse':
            data_lines = [line[5:].strip() for line in event.data.splitlines() if line.startswith('data:')]
            if data_lines and data_lines[0] != '[DONE]':
                self._consume_payload(state, json.loads('\n'.join(data_lines)), event.timestamp)
