from __future__ import annotations

import json
from typing import Any, Dict, Optional

from evalscope.perf.config.models import GenerationConfig
from evalscope.perf.domain.observation import TokenSource, TransportEvent
from evalscope.perf.domain.workload import SingleTurnItem
from evalscope.perf.protocols.base import ProtocolAdapter, ProtocolMeta, ProtocolState
from evalscope.perf.transport.base import HttpRequest


class OpenAIChatProtocol(ProtocolAdapter):
    meta = ProtocolMeta(name='openai_chat', endpoint='chat/completions')

    def build_request(
        self,
        item: SingleTurnItem,
        generation: GenerationConfig,
        max_tokens: Optional[int] = None,
    ) -> HttpRequest:
        messages = item.messages
        if isinstance(messages, str):
            messages = [{'role': 'user', 'content': messages}]
        if isinstance(messages, dict) and ('messages' in messages or 'prompt' in messages):
            payload = {'model': self.target.model, **messages}
        else:
            payload = {'model': self.target.model, 'messages': messages}
        self._apply_generation(payload, generation, max_tokens)
        if generation.stream:
            payload['stream_options'] = {'include_usage': True}
        headers = dict(self.target.headers)
        if self.target.api_key:
            headers['Authorization'] = f'Bearer {self.target.api_key}'
        return HttpRequest(url=self.url(), body=payload, headers=headers)

    @staticmethod
    def _usage(state: ProtocolState, payload: Dict[str, Any]) -> None:
        usage = payload.get('usage')
        if not isinstance(usage, dict):
            return
        state.prompt_tokens = usage.get('prompt_tokens')
        state.completion_tokens = usage.get('completion_tokens')
        details = usage.get('prompt_tokens_details') or {}
        state.cached_tokens = details.get('cached_tokens')
        state.accepted_draft_tokens = usage.get('accepted_draft_tokens')
        state.proposed_draft_tokens = usage.get('proposed_draft_tokens')
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
        self._usage(state, payload)
        choices = payload.get('choices') or []
        if not choices:
            return
        choice = choices[0]
        delta = choice.get('delta') or choice.get('message') or {}
        text = (delta.get('content') or '') + (delta.get('reasoning_content') or '')
        if not text and 'text' in choice:
            text = choice.get('text') or ''
        if text:
            if state.first_token_time is None:
                state.first_token_time = timestamp
            state.chunk_times.append(timestamp)
            state.generated_text += text

    def consume_event(self, state: ProtocolState, event: TransportEvent) -> None:
        if self._consume_common(state, event):
            return
        if event.kind == 'json':
            self._consume_payload(state, event.data, event.timestamp)
            return
        if event.kind == 'sse':
            message = event.data.strip()
            if not message or message.startswith(':'):
                return
            data_lines = [line[5:].strip() for line in message.splitlines() if line.startswith('data:')]
            if not data_lines:
                return
            raw = '\n'.join(data_lines)
            if raw == '[DONE]':
                return
            self._consume_payload(state, json.loads(raw), event.timestamp)
