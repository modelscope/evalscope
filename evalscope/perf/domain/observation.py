from __future__ import annotations

from enum import Enum
from pydantic import BaseModel, ConfigDict, Field
from typing import Any, Dict, List, Literal, Optional


class TokenSource(str, Enum):
    SERVER_REPORTED = 'server_reported'
    ESTIMATED = 'estimated'
    UNAVAILABLE = 'unavailable'


class TransportEvent(BaseModel):
    """One timestamped event emitted by an HTTP transport."""

    model_config = ConfigDict(frozen=True)

    kind: str
    timestamp: float
    data: Any = None
    status_code: Optional[int] = None
    content_type: Optional[str] = None


class RequestObservation(BaseModel):
    """Canonical raw observation for one attempted request."""

    run_id: str
    request_id: str
    trace_id: Optional[str] = None
    turn_index: Optional[int] = None
    is_first_turn: bool = False
    is_last_turn: bool = False
    is_warmup: bool = False
    scheduled_time: Optional[float] = None
    dispatch_time: Optional[float] = None
    start_time: Optional[float] = None
    first_token_time: Optional[float] = None
    completed_time: Optional[float] = None
    success: bool = False
    dropped: bool = False
    drop_reason: Optional[str] = None
    outstanding: Optional[int] = None
    status_code: Optional[int] = None
    error_type: Optional[str] = None
    error: Optional[str] = None
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    cached_tokens: Optional[int] = None
    accepted_draft_tokens: Optional[int] = None
    proposed_draft_tokens: Optional[int] = None
    prompt_token_source: TokenSource = TokenSource.UNAVAILABLE
    completion_token_source: TokenSource = TokenSource.UNAVAILABLE
    cache_token_source: TokenSource = TokenSource.UNAVAILABLE
    chunk_times: List[float] = Field(default_factory=list)
    generated_text: str = ''
    request_payload: Dict[str, Any] = Field(default_factory=dict)
    response_payloads: List[Any] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    gpu_metrics: Dict[str, float] = Field(default_factory=dict)
    gpu_location: Optional[Literal['client', 'server_reported']] = None

    @property
    def latency(self) -> Optional[float]:
        if self.start_time is None or self.completed_time is None:
            return None
        return max(0.0, self.completed_time - self.start_time)

    @property
    def ttft(self) -> Optional[float]:
        if self.start_time is None or self.first_token_time is None:
            return None
        return max(0.0, self.first_token_time - self.start_time)

    @property
    def arrival_lag(self) -> Optional[float]:
        if self.scheduled_time is None or self.dispatch_time is None:
            return None
        return max(0.0, self.dispatch_time - self.scheduled_time)

    @property
    def inter_token_latencies(self) -> List[float]:
        return [right - left for left, right in zip(self.chunk_times, self.chunk_times[1:])]

    @property
    def tpot(self) -> Optional[float]:
        if self.latency is None or self.ttft is None or not self.completion_tokens or self.completion_tokens <= 1:
            return None
        return max(0.0, (self.latency - self.ttft) / (self.completion_tokens - 1))
