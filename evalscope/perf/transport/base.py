from __future__ import annotations

from abc import ABC, abstractmethod
from pydantic import BaseModel, ConfigDict, Field
from typing import Any, AsyncIterator, Dict, Optional

from evalscope.perf.domain.observation import TransportEvent


class HttpRequest(BaseModel):
    """Protocol-independent HTTP request description."""

    model_config = ConfigDict(frozen=True)

    url: str
    body: Dict[str, Any]
    headers: Dict[str, str] = Field(default_factory=dict)


class HttpTransport(ABC):
    """Transport interface yielding timestamped HTTP events."""

    @abstractmethod
    async def __aenter__(self) -> 'HttpTransport':
        raise NotImplementedError

    @abstractmethod
    async def __aexit__(self, exc_type, exc, tb) -> None:
        raise NotImplementedError

    @abstractmethod
    async def events(self, request: HttpRequest) -> AsyncIterator[TransportEvent]:
        if False:
            yield  # pragma: no cover
