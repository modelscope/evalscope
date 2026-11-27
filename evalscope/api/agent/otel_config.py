"""Configuration for importing OpenTelemetry GenAI traces.

Lives next to :mod:`evalscope.api.agent.otel` (not under ``agent/external``):
the ``external`` package means "run an external CLI", whereas importing a trace
executes nothing at all — it is the pull side of the same data.
"""

from typing import Literal

from pydantic import Field

from evalscope.api.agent.types import BaseAgentConfig

__all__ = ['OTelTraceConfig']


class OTelTraceConfig(BaseAgentConfig):
    """Carried by ``TaskConfig.agent_config`` when ``mode == 'otel_trace'``.

    Imports an existing OpenTelemetry GenAI trace export as a scorable sample
    instead of running an agent: the importer reconstructs the ``AgentTrace``
    and message history the external-agent bridge would have produced.
    """

    mode: Literal['otel_trace'] = Field(default='otel_trace')
    """Union discriminator — imports a trace rather than executing an agent."""

    trace_path: str
    """Path to an OTLP JSON export, or a directory of ``<sample_id>.otlp.json`` files."""

    sample_id_attribute: str = Field(default='evalscope.sample_id')
    """Span attribute carrying the eval sample id when several samples share one export."""

    subset_attribute: str = Field(default='evalscope.subset')
    """Span attribute carrying the eval subset name.

    A sample id is unique only within its subset, so an export that serves
    several subsets must also tag the subset; otherwise the binding is
    ambiguous and the import is rejected.
    """
