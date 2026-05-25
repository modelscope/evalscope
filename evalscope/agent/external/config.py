"""Pydantic configuration for external-agent runs.

Lives under the shared :class:`BaseAgentConfig` so ``environment`` /
``environment_extra`` are not duplicated against :class:`NativeAgentConfig`.
The ``mode`` literal serves as the Pydantic discriminator on the
:attr:`TaskConfig.agent_config` union.
"""

from pydantic import BaseModel, Field, field_validator
from typing import Literal, Optional

from evalscope.api.agent.types import BaseAgentConfig


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

    CODEX = 'codex'
    """OpenAI's official ``codex exec`` CLI (uses the Responses API)."""


class BridgeConfig(BaseModel):
    """Knobs for the reverse-proxy bridge that sits between an external
    agent CLI and EvalScope's model layer.
    """

    proxy_host: str = Field(default='127.0.0.1')
    """Host the proxy binds to.  ``0.0.0.0`` is required when the agent
    runs in a container that needs to dial back via ``host.docker.internal``;
    ``127.0.0.1`` is fine for ``LocalAgentEnvironment``."""

    proxy_port: Optional[int] = Field(default=None)
    """Port the proxy binds to.  ``None`` lets the OS pick a free port —
    recommended unless you need a stable URL."""


class ExternalAgentConfig(BaseAgentConfig):
    """Carried by ``TaskConfig.agent_config`` when ``mode == 'external'``.

    Every ``DefaultDataAdapter``-based benchmark routes inference through
    an external agent CLI driven by :class:`AgentRunner`, instead of
    issuing a single ``model.generate`` call.
    """

    mode: Literal['external'] = Field(default='external')
    """Union discriminator — fixed value for the external-agent path."""

    framework: str
    """Registered runner name; see :class:`ExternalAgentFramework` for the
    built-in set.  Validated at construction time against
    :data:`evalscope.api.registry.RUNNER_REGISTRY` — unknown names fail
    fast with an "Available: [...]" hint."""

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
