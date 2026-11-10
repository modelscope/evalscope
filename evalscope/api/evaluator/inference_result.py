"""Typed return value for ``_on_inference`` hooks that produce agent state.

Non-agent benchmarks return a plain :class:`ModelOutput` from
``_on_inference``. Agent benchmarks (custom multi-turn drivers, AgentLoop,
BFCL, tau_bench, ...) additionally produce a full message trajectory and
optionally a structured trace that need to flow into :class:`TaskState` for
review/visualization.
"""

from dataclasses import dataclass
from typing import List, Optional, Union

from evalscope.api.messages import ChatMessage
from evalscope.api.model import ModelOutput


@dataclass
class InferenceResult:
    """Bundle of model output plus optional agent trajectory state.

    ``_on_inference`` may return either a bare :class:`ModelOutput` (the
    classic single-turn contract) or this richer structure when there is a
    multi-turn message history and/or structured trace to surface on the
    resulting :class:`TaskState`.
    """

    output: ModelOutput
    """Final ``ModelOutput`` (the prediction artifact)."""

    messages: Optional[List[ChatMessage]] = None
    """Full conversation messages; overrides ``TaskState.messages`` when set."""

    trace: Optional[object] = None
    """Optional structured agent trace assigned to ``TaskState.agent_trace``.

    Typed as ``object`` to avoid an import cycle with ``api.agent``;
    concrete callers pass an ``AgentTrace`` instance.
    """


InferenceReturn = Union[ModelOutput, InferenceResult]
"""Allowed return type from ``DefaultDataAdapter._on_inference``."""
