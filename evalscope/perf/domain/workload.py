from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field
from typing import Any, Dict, List, Literal, Optional, Union


class SingleTurnItem(BaseModel):
    """One independent request input produced by a workload."""

    model_config = ConfigDict(frozen=True)

    messages: Union[str, List[str], List[Dict[str, Any]], List[int], Dict[str, Any]]
    metadata: Dict[str, Any] = Field(default_factory=dict)


class Turn(BaseModel):
    """One user/tool delta in a conversation workload."""

    model_config = ConfigDict(frozen=True)

    messages: List[Dict[str, Any]]
    max_tokens: Optional[int] = None
    tool_call_latency: Optional[float] = None


class ConversationItem(BaseModel):
    """A complete conversation whose turns must run sequentially."""

    model_config = ConfigDict(frozen=True)

    turns: List[Turn]
    metadata: Dict[str, Any] = Field(default_factory=dict)


WorkItem = Union[SingleTurnItem, ConversationItem]
WorkloadMode = Literal['single_turn', 'conversation']


class WorkloadMeta(BaseModel):
    """Static capabilities declared by a workload plugin."""

    model_config = ConfigDict(frozen=True)

    name: str
    mode: WorkloadMode
    requires_dataset: bool = True
    supports_parallel_generation: bool = False
    supports_local_path: bool = False
    supports_modelscope: bool = False
    supports_huggingface: bool = False
    requires_tokenizer: bool = False
    protocols: frozenset[str] = frozenset({'openai_chat'})
