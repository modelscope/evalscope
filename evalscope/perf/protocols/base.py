from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pydantic import BaseModel, ConfigDict
from typing import Any, Dict, List, Optional

from evalscope.perf.config.models import GenerationConfig, TargetConfig
from evalscope.perf.domain.observation import TokenSource, TransportEvent
from evalscope.perf.domain.workload import SingleTurnItem
from evalscope.perf.transport.base import HttpRequest


@dataclass
class ProtocolState:
    request: HttpRequest
    start_time: Optional[float] = None
    first_token_time: Optional[float] = None
    completed_time: Optional[float] = None
    status_code: Optional[int] = None
    success: bool = False
    error_type: Optional[str] = None
    error: Optional[str] = None
    generated_text: str = ''
    response_payloads: List[Any] = field(default_factory=list)
    chunk_times: List[float] = field(default_factory=list)
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    cached_tokens: Optional[int] = None
    accepted_draft_tokens: Optional[int] = None
    proposed_draft_tokens: Optional[int] = None
    prompt_token_source: TokenSource = TokenSource.UNAVAILABLE
    completion_token_source: TokenSource = TokenSource.UNAVAILABLE
    cache_token_source: TokenSource = TokenSource.UNAVAILABLE


@dataclass(frozen=True)
class ProtocolResult:
    start_time: Optional[float]
    first_token_time: Optional[float]
    completed_time: Optional[float]
    status_code: Optional[int]
    success: bool
    error_type: Optional[str]
    error: Optional[str]
    generated_text: str
    response_payloads: List[Any]
    chunk_times: List[float]
    prompt_tokens: Optional[int]
    completion_tokens: Optional[int]
    cached_tokens: Optional[int]
    accepted_draft_tokens: Optional[int]
    proposed_draft_tokens: Optional[int]
    prompt_token_source: TokenSource
    completion_token_source: TokenSource
    cache_token_source: TokenSource


class ProtocolMeta(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    endpoint: str


class ProtocolAdapter(ABC):
    """Encode workload items and consume protocol-specific response events."""

    meta: ProtocolMeta

    def __init__(self, target: TargetConfig) -> None:
        self.target = target
        self.tokenizer = None
        if target.tokenizer:
            from evalscope.perf.workloads.builtins.tokenizer import load_tokenizer

            self.tokenizer = load_tokenizer(target.tokenizer)

    def url(self) -> str:
        base = self.target.base_url.rstrip('/')
        known = ('chat/completions', 'completions', 'responses', 'embeddings', 'reranks')
        if any(base.endswith('/' + suffix) for suffix in known):
            return base
        return f'{base}/{self.meta.endpoint}'

    @abstractmethod
    def build_request(
        self,
        item: SingleTurnItem,
        generation: GenerationConfig,
        max_tokens: Optional[int] = None,
    ) -> HttpRequest:
        raise NotImplementedError

    @abstractmethod
    def consume_event(self, state: ProtocolState, event: TransportEvent) -> None:
        raise NotImplementedError

    def new_state(self, request: HttpRequest) -> ProtocolState:
        return ProtocolState(request=request)

    def finalize(self, state: ProtocolState) -> ProtocolResult:
        """Fill missing token counts with an explicitly estimated fallback."""
        if self.tokenizer is None:
            return self._result(state)
        if state.prompt_tokens is None:
            raw = json.dumps(state.request.body, ensure_ascii=False)
            state.prompt_tokens = len(self.tokenizer.encode(raw, add_special_tokens=False))
            state.prompt_token_source = TokenSource.ESTIMATED
        if state.completion_tokens is None:
            state.completion_tokens = len(self.tokenizer.encode(state.generated_text, add_special_tokens=False))
            state.completion_token_source = TokenSource.ESTIMATED
        return self._result(state)

    @staticmethod
    def _result(state: ProtocolState) -> ProtocolResult:
        return ProtocolResult(
            start_time=state.start_time,
            first_token_time=state.first_token_time,
            completed_time=state.completed_time,
            status_code=state.status_code,
            success=state.success,
            error_type=state.error_type,
            error=state.error,
            generated_text=state.generated_text,
            response_payloads=list(state.response_payloads),
            chunk_times=list(state.chunk_times),
            prompt_tokens=state.prompt_tokens,
            completion_tokens=state.completion_tokens,
            cached_tokens=state.cached_tokens,
            accepted_draft_tokens=state.accepted_draft_tokens,
            proposed_draft_tokens=state.proposed_draft_tokens,
            prompt_token_source=state.prompt_token_source,
            completion_token_source=state.completion_token_source,
            cache_token_source=state.cache_token_source,
        )

    def _consume_common(self, state: ProtocolState, event: TransportEvent) -> bool:
        if event.kind == 'request_start':
            state.start_time = event.timestamp
            return True
        if event.kind == 'response_start':
            state.status_code = event.status_code
            return True
        if event.kind == 'http_error':
            state.status_code = event.status_code
            state.error_type = 'http_error'
            state.error = json.dumps(event.data, ensure_ascii=False) if not isinstance(event.data, str) else event.data
            state.completed_time = event.timestamp
            return True
        if event.kind == 'response_end':
            state.completed_time = event.timestamp
            state.success = state.error is None
            return True
        return False

    @staticmethod
    def _apply_generation(payload: Dict[str, Any], generation: GenerationConfig, max_tokens: Optional[int]) -> None:
        payload['max_tokens'] = max_tokens if max_tokens is not None else generation.max_tokens
        payload['temperature'] = generation.temperature
        payload['stream'] = generation.stream
        optional = {
            'min_tokens': generation.min_tokens,
            'top_p': generation.top_p,
            'top_k': generation.top_k,
            'frequency_penalty': generation.frequency_penalty,
            'repetition_penalty': generation.repetition_penalty,
            'logprobs': generation.logprobs,
            'n': generation.n_choices,
            'stop': generation.stop,
            'stop_token_ids': generation.stop_token_ids,
        }
        payload.update({key: value for key, value in optional.items() if value is not None})
        payload.update(generation.extra)
