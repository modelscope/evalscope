"""Pydantic configuration for external-agent runs.

Mirrors the design in ``.qoder/plans/agent_bridge_design.md`` §7.2.
P0 keeps the shape minimal (only fields actually consumed); fairness levels
(L2 / L3 overrides) are accepted but not enforced yet.
"""

from pydantic import BaseModel, Field, field_validator
from typing import Any, Dict, Literal, Optional


class ExternalAgentFramework:
    """Built-in external-agent runner names.

    Third-party runners can register additional names via
    :func:`evalscope.api.registry.register_runner`; this class enumerates
    only the runners shipped with EvalScope so IDEs and CLI users have a
    discoverable surface (mirrors :class:`evalscope.constants.EvalType`).
    """

    MOCK = 'mock'
    """One-shot Anthropic client used for bridge smoke tests."""

    CLAUDE_CODE = 'claude-code'
    """Anthropic's official ``claude --print`` CLI."""


class BridgeConfig(BaseModel):
    """Knobs for the reverse-proxy bridge that sits between an external
    agent CLI and EvalScope's model layer.
    """

    override_mode: Literal['L1', 'L2', 'L3'] = Field(default='L1')
    """Fairness level.  P0 implements L1 only (no overrides) — L2 / L3 are
    accepted for forward compatibility and logged as ``not yet enforced``.
    """

    record_trajectory: bool = Field(default=True)
    """When True the bridge records every request/response into a
    ``Trajectory`` snapshot attached to the sample metadata."""

    proxy_host: str = Field(default='127.0.0.1')
    """Host the proxy binds to.  ``0.0.0.0`` is required when the agent
    runs in a container that needs to dial back via ``host.docker.internal``;
    ``127.0.0.1`` is fine for ``LocalAgentEnvironment``."""

    proxy_port: Optional[int] = Field(default=None)
    """Port the proxy binds to.  ``None`` lets the OS pick a free port —
    recommended unless you need a stable URL."""


class ExternalAgentConfig(BaseModel):
    """Carried by ``TaskConfig.external_agent``.

    When set, every ``DefaultDataAdapter``-based benchmark routes inference
    through an external agent CLI driven by :class:`AgentRunner`, instead
    of issuing a single ``model.generate`` call.
    """

    framework: str
    """Registered runner name; see :class:`ExternalAgentFramework` for the
    built-in set.  Validated at construction time against
    :data:`evalscope.api.registry.RUNNER_REGISTRY` — unknown names fail
    fast with an "Available: [...]" hint."""

    kwargs: Dict[str, Any] = Field(default_factory=dict)
    """Runner-specific keyword arguments forwarded to the runner ctor."""

    environment: str = Field(default='local')
    """Registered environment name (see
    :data:`evalscope.api.registry.ENVIRONMENT_REGISTRY`).  Built-ins:
    ``'local'`` / ``'enclave'`` / ``'docker'`` / ``'volcengine'``.
    Mirrors :attr:`evalscope.api.agent.AgentConfig.environment`."""

    environment_extra: Dict[str, Any] = Field(default_factory=dict)
    """Kwargs forwarded verbatim to the environment constructor (e.g.
    ``working_dir``, ``env_vars`` for ``local``; ``engine``,
    ``sandbox_config``, ``manager_config``, ``timeout`` for ``enclave``).
    Mirrors :attr:`evalscope.api.agent.AgentConfig.environment_extra`."""

    bridge: BridgeConfig = Field(default_factory=BridgeConfig)
    """Bridge-level options."""

    timeout: Optional[float] = Field(default=600.0)
    """Per-sample wall-clock budget passed to the runner."""

    @field_validator('framework')
    @classmethod
    def _validate_framework(cls, v: str) -> str:
        from evalscope.api.registry import get_runner
        get_runner(v)  # raises ValueError with available list on typo
        return v
