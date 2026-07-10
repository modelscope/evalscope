from __future__ import annotations

from typing import Any, Dict, Optional

from evalscope.perf.config.models import GenerationConfig
from evalscope.perf.domain.observation import TokenSource, TransportEvent
from evalscope.perf.domain.workload import SingleTurnItem
from evalscope.perf.protocols.base import ProtocolAdapter, ProtocolMeta, ProtocolState
from evalscope.perf.transport.base import HttpRequest


class OpenAIEmbeddingProtocol(ProtocolAdapter):
    meta = ProtocolMeta(name='openai_embedding', endpoint='embeddings')

    def build_request(
        self,
        item: SingleTurnItem,
        generation: GenerationConfig,
        max_tokens: Optional[int] = None,
    ) -> HttpRequest:
        payload: Dict[str, Any] = {'model': self.target.model, 'input': item.messages}
        payload.update(generation.extra)
        headers = dict(self.target.headers)
        if self.target.api_key:
            headers['Authorization'] = f'Bearer {self.target.api_key}'
        return HttpRequest(url=self.url(), body=payload, headers=headers)

    def consume_event(self, state: ProtocolState, event: TransportEvent) -> None:
        if self._consume_common(state, event):
            return
        if event.kind != 'json':
            return
        payload = event.data
        state.response_payloads.append(payload)
        if isinstance(payload, dict):
            usage = payload.get('usage') or {}
            state.prompt_tokens = usage.get('prompt_tokens', usage.get('total_tokens'))
            state.completion_tokens = 0
            if usage:
                state.prompt_token_source = TokenSource.SERVER_REPORTED
                state.completion_token_source = TokenSource.SERVER_REPORTED
            data = payload.get('data') or []
            if data:
                state.generated_text = f'embedding_count={len(data)}'
        state.first_token_time = event.timestamp
        state.chunk_times.append(event.timestamp)
