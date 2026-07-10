from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterator

from evalscope.perf.domain.observation import RequestObservation


class ResultStore(ABC):

    @abstractmethod
    def open(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def write(self, observation: RequestObservation) -> None:
        raise NotImplementedError

    @abstractmethod
    def observations(self, include_warmup: bool = False) -> Iterator[RequestObservation]:
        raise NotImplementedError

    @abstractmethod
    def close(self) -> None:
        raise NotImplementedError
