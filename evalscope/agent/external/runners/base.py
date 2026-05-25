"""Runner protocol shared by every external-agent integration.

A runner's only job is to (1) install / set up the agent CLI inside the
sandbox and (2) launch one run that talks to the bridge.  The bridge
captures the trajectory; the runner returns the raw stdout the CLI
emitted.
"""

from abc import ABC, abstractmethod
from pydantic import BaseModel, Field
from typing import Any, Dict, Optional

from evalscope.api.agent import AgentEnvironment


class BridgeEndpoint(BaseModel):
    """Bridge connection info handed to the runner for env-var injection."""

    base_url: str
    """HTTP base URL reachable from inside the sandbox (no trailing slash)."""

    trial_token: str
    """Bearer token the bridge uses to route requests back to the trial's
    :class:`TrialSession` (format: ``trial-<hex>``)."""


class RunnerTimeoutError(RuntimeError):
    """Raised by an :class:`AgentRunner` when the wrapped CLI exceeded its
    ``task.timeout`` budget.  The adapter catches this distinctly so the
    bridge ``RUN_END`` event can record ``timed_out=True`` instead of
    masking it as a generic runtime failure.
    """


class ExternalAgentTask(BaseModel):
    """One unit of work passed to :meth:`AgentRunner.run`."""

    instruction: str
    """Natural-language task description forwarded to the agent CLI."""

    timeout: Optional[float] = Field(default=None)
    """Per-sample wall-clock budget; runner enforces via ``env.exec(timeout=...)``."""

    metadata: Dict[str, Any] = Field(default_factory=dict)
    """Free-form passthrough (sample id, benchmark hints, ...)."""


class AgentRunResult(BaseModel):
    """Aggregate returned by :meth:`AgentRunner.run`.

    ``trajectory`` is intentionally left as ``None`` here — the adapter
    layer pulls it from the bridge's recorder after the run finishes.
    """

    output: str
    """Final answer string (typically the agent CLI's stdout)."""

    metrics: Dict[str, Any] = Field(default_factory=dict)
    """Wall-time, exit code, runner-specific counters."""


class AgentRunner(ABC):
    """Drive one external-agent run inside an :class:`AgentEnvironment`."""

    framework: str = 'base'
    """Registered runner name (used by ``ExternalAgentConfig.framework``)."""

    @abstractmethod
    async def setup(self, env: AgentEnvironment) -> None:
        """Install the agent CLI / dependencies inside ``env``.  Called once
        per sample before :meth:`run`.  May be a no-op for pre-baked images."""

    @abstractmethod
    async def run(
        self,
        task: ExternalAgentTask,
        env: AgentEnvironment,
        bridge: BridgeEndpoint,
    ) -> AgentRunResult:
        """Execute the agent CLI for one sample and return its raw output."""
