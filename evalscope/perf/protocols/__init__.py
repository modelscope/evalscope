from typing import Dict, List, Type

from evalscope.perf.config.models import TargetConfig
from evalscope.perf.domain.errors import PerfConfigError
from evalscope.perf.protocols.base import ProtocolAdapter, ProtocolResult, ProtocolState
from evalscope.perf.protocols.openai_chat import OpenAIChatProtocol
from evalscope.perf.protocols.openai_completions import OpenAICompletionsProtocol
from evalscope.perf.protocols.openai_embedding import OpenAIEmbeddingProtocol
from evalscope.perf.protocols.openai_rerank import OpenAIRerankProtocol
from evalscope.perf.protocols.openai_responses import OpenAIResponsesProtocol


class ProtocolRegistry:

    def __init__(self) -> None:
        self._classes: Dict[str, Type[ProtocolAdapter]] = {}

    def register(self, adapter: Type[ProtocolAdapter]) -> Type[ProtocolAdapter]:
        name = adapter.meta.name
        if name in self._classes:
            raise PerfConfigError(f'Protocol {name!r} is already registered')
        self._classes[name] = adapter
        return adapter

    def get(self, name: str) -> Type[ProtocolAdapter]:
        if name not in self._classes:
            raise PerfConfigError(f'Unknown protocol {name!r}. Available protocols: {", ".join(self.names())}')
        return self._classes[name]

    def names(self) -> List[str]:
        return sorted(self._classes)


protocol_registry = ProtocolRegistry()
for _adapter in (
    OpenAIChatProtocol,
    OpenAICompletionsProtocol,
    OpenAIResponsesProtocol,
    OpenAIEmbeddingProtocol,
    OpenAIRerankProtocol,
):
    protocol_registry.register(_adapter)


def register_protocol(adapter: Type[ProtocolAdapter]) -> Type[ProtocolAdapter]:
    return protocol_registry.register(adapter)


def create_protocol(target: TargetConfig) -> ProtocolAdapter:
    return protocol_registry.get(target.protocol)(target)


__all__ = [
    'ProtocolAdapter',
    'ProtocolRegistry',
    'ProtocolResult',
    'ProtocolState',
    'create_protocol',
    'protocol_registry',
    'register_protocol',
]
