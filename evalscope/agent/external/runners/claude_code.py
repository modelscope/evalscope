"""Runner for Anthropic's ``claude-code`` CLI.

Points ``claude --print`` at the bridge via ``ANTHROPIC_BASE_URL`` /
``ANTHROPIC_API_KEY`` env vars and returns whatever the CLI prints.  The
CLI must be on the sandbox PATH — installation is left to the environment
(``npm install -g @anthropic-ai/claude-code`` on Local; pre-baked into
the image on Enclave).

Important: when the host ``HOME`` contains a logged-in claude-code OAuth
token (typical dev box), the CLI prefers the keychain over
``ANTHROPIC_BASE_URL`` env vars and silently bypasses the bridge.  We
default to a fresh ``HOME`` per run to force env-var-driven routing.
"""

import tempfile
from typing import Any, Dict, List, Optional

from evalscope.api.agent import AgentEnvironment
from evalscope.api.registry import register_runner
from evalscope.utils.logger import get_logger
from .base import AgentRunner, AgentRunResult, BridgeEndpoint, ExternalAgentTask

logger = get_logger()


@register_runner('claude-code')
class ClaudeCodeRunner(AgentRunner):
    """Drive ``claude --print`` for one sample.

    Kwargs forwarded from ``ExternalAgentConfig.kwargs``:

    * ``model_name``     — overrides the model claude-code dials (forwarded
      to the bridge inside the request body).  Defaults to whatever the
      evalscope-side Model uses.
    * ``allowed_tools``  — passed to ``--allowedTools``.  ``''`` disables
      all tools (useful for math / single-shot evals); ``None`` leaves
      claude-code's defaults intact.
    * ``disallowed_tools`` — passed to ``--disallowedTools``.
    * ``skip_permissions`` — when True (default), passes
      ``--dangerously-skip-permissions`` so the CLI does not prompt.
    * ``bare`` — when True passes ``--bare`` for the lightest-weight
      execution (skips hooks, CLAUDE.md autodiscovery, etc.).  Defaults
      to False so user CLAUDE.md / settings still apply.
    * ``extra_args`` — list of additional CLI arguments appended verbatim
      before the prompt.
    * ``home_override`` — optional path used as ``HOME`` for the subprocess.
      Defaults to a fresh per-run tempdir so the CLI cannot pick up a
      logged-in OAuth token from the host keychain and bypass the bridge.
      Pass an explicit path to reuse user settings, or ``''`` to inherit
      the parent shell's ``HOME``.
    """

    framework: str = 'claude-code'

    def __init__(
        self,
        *,
        model_name: str = '',
        allowed_tools: Optional[str] = None,
        disallowed_tools: Optional[str] = None,
        skip_permissions: bool = True,
        bare: bool = False,
        extra_args: Optional[List[str]] = None,
        home_override: Optional[str] = None,
        **_: Any,
    ) -> None:
        self._model_name = model_name
        self._allowed_tools = allowed_tools
        self._disallowed_tools = disallowed_tools
        self._skip_permissions = skip_permissions
        self._bare = bare
        self._extra_args = list(extra_args or [])
        self._home_override = home_override

    async def setup(self, env: AgentEnvironment) -> None:
        """Verify the ``claude`` CLI is reachable inside ``env``.

        Installation is out of scope for P0 — the CLI must already be on
        PATH (locally or pre-baked into the sandbox image).
        """
        probe = await env.exec(['bash', '-c', 'command -v claude && claude --version'])
        if probe.returncode != 0:
            raise RuntimeError(
                'claude CLI not found in the agent environment. Install it with '
                '`npm install -g @anthropic-ai/claude-code` (or bake into the sandbox image).'
                f' stderr={probe.stderr!r}'
            )
        logger.debug(f'claude-code probe: {probe.stdout.strip()!r}')

    async def run(
        self,
        task: ExternalAgentTask,
        env: AgentEnvironment,
        bridge: BridgeEndpoint,
    ) -> AgentRunResult:
        env_vars: Dict[str, str] = {
            # ``ANTHROPIC_BASE_URL`` must be the root URL — the Anthropic
            # SDK appends ``/v1/messages`` itself, and matches the bridge's
            # ``/anthropic/v1/messages`` route via the ``/anthropic`` segment.
            'ANTHROPIC_BASE_URL': f'{bridge.base_url}/anthropic',
            # Set both auth variants so the CLI uses whichever it prefers.
            'ANTHROPIC_API_KEY': bridge.trial_token,
            'ANTHROPIC_AUTH_TOKEN': bridge.trial_token,
            # Anthropic SDK's "model" env var (used by some auto-discovery paths).
            **({
                'ANTHROPIC_MODEL': self._model_name
            } if self._model_name else {}),
            # Inspect-AI's two load-bearing env vars: suppress the OAuth /
            # keychain probe and the telemetry / auto-update HTTPS calls
            # that otherwise block for ~60s in offline / bridged setups.
            'IS_SANDBOX': '1',
            'CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC': '1',
        }
        home_dir = self._resolve_home()
        if home_dir is not None:
            env_vars['HOME'] = home_dir

        # Pass the prompt as the trailing positional argument (matches
        # claude-code's documented invocation pattern).  Avoid variadic
        # flags like ``--allowedTools <tools...>`` before the positional
        # because they would consume the prompt as a tool value.
        cmd: List[str] = ['claude', '--print', '--no-session-persistence', '--output-format', 'text']
        if self._bare:
            cmd.append('--bare')
        if self._skip_permissions:
            cmd.append('--dangerously-skip-permissions')
        if self._model_name:
            cmd.extend(['--model', self._model_name])
        if self._allowed_tools is not None:
            cmd.extend(['--allowedTools', self._allowed_tools])
        if self._disallowed_tools is not None:
            cmd.extend(['--disallowedTools', self._disallowed_tools])
        cmd.extend(self._extra_args)
        cmd.append(task.instruction)

        result = await env.exec(
            cmd,
            cwd=task.cwd,
            timeout=task.timeout,
            env=env_vars,
        )
        if result.returncode != 0 or result.timed_out:
            tail_stderr = (result.stderr or '').strip()[-2000:]
            raise RuntimeError(
                f'claude-code exited with code {result.returncode} '
                f'(timed_out={result.timed_out}): {tail_stderr}'
            )
        return AgentRunResult(
            output=result.stdout.strip(),
            metrics={
                'wall_time': result.duration,
                'returncode': result.returncode,
            },
        )

    def _resolve_home(self) -> Optional[str]:
        """Pick the ``HOME`` value used for the subprocess.

        ``None`` returned by this method means "inherit parent HOME"; an
        empty-string user override also inherits.  Any other string is
        used verbatim.  ``None`` from the field (the default) creates a
        fresh tempdir per call so the CLI cannot read a logged-in OAuth
        token from the host keychain.
        """
        if self._home_override == '':
            return None  # explicit inherit
        if self._home_override is not None:
            return self._home_override
        return tempfile.mkdtemp(prefix='evalscope-claude-code-')
