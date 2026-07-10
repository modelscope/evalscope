from __future__ import annotations

import json
from typing import Any, Dict, Optional

from evalscope.perf.config.models import GenerationConfig
from evalscope.perf.domain.observation import TokenSource, TransportEvent
from evalscope.perf.domain.workload import SingleTurnItem
from evalscope.perf.protocols.base import ProtocolAdapter, ProtocolMeta, ProtocolState
from evalscope.perf.transport.base import HttpRequest


class OpenAICompletionsProtocol(ProtocolAdapter):
    """Canonical adapter for the OpenAI-compatible text completions endpoint."""

    meta = ProtocolMeta(name='openai_completions', endpoint='completions')

    def build_request(
        self,
        item: SingleTurnItem,
        generation: GenerationConfig,
        max_tokens: Optional[int] = None,
    ) -> HttpRequest:
        if isinstance(item.messages, dict):
            payload: Dict[str, Any] = {'model': self.target.model, **item.messages}
        else:
            payload = {'model': self.target.model, 'prompt': item.messages}
        self._apply_generation(payload, generation, max_tokens)
        if generation.stream:
            payload['stream_options'] = {'include_usage': True}
        headers = dict(self.target.headers)
        if self.target.api_key:
            headers['Authorization'] = f'Bearer {self.target.api_key}'
        return HttpRequest(url=self.url(), body=payload, headers=headers)

    @staticmethod
    def _usage(state: ProtocolState, payload: Dict[str, Any]) -> None:
        usage = payload.get('usage') or {}
        state.prompt_tokens = usage.get('prompt_tokens')
        state.completion_tokens = usage.get('completion_tokens')
        details = usage.get('prompt_tokens_details') or {}
        state.cached_tokens = details.get('cached_tokens')
        if state.prompt_tokens is not None:
            state.prompt_token_source = TokenSource.SERVER_REPORTED
        if state.completion_tokens is not None:
            state.completion_token_source = TokenSource.SERVER_REPORTED
        if state.cached_tokens is not None:
            state.cache_token_source = TokenSource.SERVER_REPORTED

    def _payload(self, state: ProtocolState, payload: Any, timestamp: float) -> None:
        state.response_payloads.append(payload)
        if not isinstance(payload, dict):
            return
        self._usage(state, payload)
        choices = payload.get('choices') or []
        text = choices[0].get('text') or '' if choices else ''
        if text:
            if state.first_token_time is None:
                state.first_token_time = timestamp
            state.chunk_times.append(timestamp)
            state.generated_text += text

    def consume_event(self, state: ProtocolState, event: TransportEvent) -> None:
        if self._consume_common(state, event):
            return
        if event.kind == 'json':
            self._payload(state, event.data, event.timestamp)
        elif event.kind == 'sse':
            data = [line[5:].strip() for line in event.data.splitlines() if line.startswith('data:')]
            if data and data[0] != '[DONE]':
                self._payload(state, json.loads('\n'.join(data)), event.timestamp)
