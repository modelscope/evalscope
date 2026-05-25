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

import shutil
import tempfile
from typing import Any, Dict, List, Optional

from evalscope.api.agent import AgentEnvironment
from evalscope.api.registry import register_runner
from evalscope.utils.logger import get_logger
from .base import AgentRunner, AgentRunResult, BridgeEndpoint, ExternalAgentTask, RunnerTimeoutError

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
    * ``auto_install`` — when True (default), apt+nodesource+npm installs
      Node.js + ``@anthropic-ai/claude-code`` if ``claude --version``
      fails on probe. Set False for pre-baked images.
    * ``install_timeout_s`` — per-step wall-clock budget (default 300s)
      for the apt / nodesource / npm install commands. Bump on slow
      networks or when the npm registry / GitHub CDN is rate-limiting.
    """

    framework: str = 'claude-code'

    #: Default wall-clock budget (seconds) per install step. Cold apt+nodesource
    #: can take 2+ min; past this, assume the image is broken. Overridable
    #: per-instance via the ``install_timeout_s`` kwarg.
    _INSTALL_TIMEOUT_S: float = 300.0

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
        auto_install: bool = True,
        install_timeout_s: float = _INSTALL_TIMEOUT_S,
        node_setup_url: str = 'https://deb.nodesource.com/setup_20.x',
        npm_package: str = '@anthropic-ai/claude-code',
        **_: Any,
    ) -> None:
        self._model_name = model_name
        self._allowed_tools = allowed_tools
        self._disallowed_tools = disallowed_tools
        self._skip_permissions = skip_permissions
        self._bare = bare
        self._extra_args = list(extra_args or [])
        self._home_override = home_override
        self._auto_install = auto_install
        self._install_timeout_s = install_timeout_s
        self._node_setup_url = node_setup_url
        self._npm_package = npm_package

    async def setup(self, env: AgentEnvironment) -> None:
        """Make sure the ``claude`` CLI is reachable inside ``env``.

        First probes ``claude --version``; if the binary is already on
        PATH (pre-baked image case) we are done. Otherwise — and only
        when ``auto_install=True`` — install Node.js + the
        ``@anthropic-ai/claude-code`` npm package via apt + nodesource.

        The install path targets Debian / Ubuntu derivatives (covers
        SWE-bench / SWE-bench_Pro images). Non-apt images (Alpine,
        rhel-family) need either a pre-baked claude binary or
        ``auto_install=False`` plus a custom prep step, and will surface
        a clear error from this method rather than silently mis-running.
        """
        if await self._claude_present(env):
            return
        if not self._auto_install:
            raise RuntimeError(
                'claude CLI not found in the agent environment and auto_install=False. '
                'Either bake claude-code into the image or pass auto_install=True.'
            )
        await self._install_claude_code(env)
        if not await self._claude_present(env):
            raise RuntimeError(
                'claude CLI install reported success but `claude --version` still fails. '
                'Inspect the install logs above for the underlying cause.'
            )

    async def _claude_present(self, env: AgentEnvironment) -> bool:
        probe = await env.exec(['bash', '-c', 'command -v claude && claude --version'])
        if probe.returncode == 0:
            logger.debug(f'claude-code probe: {probe.stdout.strip()!r}')
            return True
        return False

    async def _install_claude_code(self, env: AgentEnvironment) -> None:
        """Install Node.js (when missing) and the claude-code npm package.

        Skips the Node install when ``node`` / ``npm`` are already on
        PATH — lets users opt into Node-preinstalled base images (e.g.
        ``node:20-slim``) to avoid the apt + nodesource detour and the
        in-container DNS dependency it implies.

        Each command runs as a single ``bash -c`` so apt's locking and
        ``set -e`` semantics are honoured. The combined script is also
        idempotent on retry: ``apt-get install`` short-circuits when the
        package is present, and ``npm install -g`` no-ops when the
        package version already matches.
        """
        if not await self._node_present(env):
            await self._install_node_via_apt(env)
        # Discard npm's stdout (multi-MB on cold installs); keep stderr for diagnostics.
        npm = await env.exec(
            ['bash', '-c', f'set -e; npm install -g --no-fund --no-audit {self._npm_package} >/dev/null'],
            timeout=self._install_timeout_s,
        )
        if npm.returncode != 0:
            raise RuntimeError(
                f'ClaudeCodeRunner.setup: `npm install -g {self._npm_package}` failed '
                f'(rc={npm.returncode}). stderr={npm.stderr.strip()[-1000:]!r}'
            )

    async def _node_present(self, env: AgentEnvironment) -> bool:
        probe = await env.exec(['bash', '-c', 'command -v node && command -v npm'])
        return probe.returncode == 0

    async def _install_node_via_apt(self, env: AgentEnvironment) -> None:
        """Install Node.js via nodesource (Debian / Ubuntu only path).

        Two-step: first the prerequisite apt packages, then the
        nodesource setup script + ``apt-get install nodejs``. Surfaces
        explicit errors when the image is not apt-based or has no
        network to reach the Ubuntu / Debian / nodesource mirrors.
        """
        logger.info(
            f'ClaudeCodeRunner.setup: installing Node.js via {self._node_setup_url} '
            f'(this is a one-shot per sample today; planned npm cache volume '
            f'will amortise the cost across samples).'
        )
        # Step 1: prerequisites for the nodesource installer (curl + gnupg).
        prep = await env.exec(
            [
                'bash', '-c', 'set -e; export DEBIAN_FRONTEND=noninteractive; '
                'apt-get update -qq && '
                'apt-get install -y --no-install-recommends curl ca-certificates gnupg'
            ],
            timeout=self._install_timeout_s,
        )
        if prep.returncode != 0:
            raise RuntimeError(
                f'ClaudeCodeRunner.setup: apt prerequisite install failed (rc={prep.returncode}). '
                f'This runner currently expects a Debian/Ubuntu-based image with network access, '
                f'or a base image where Node.js is already installed (e.g. node:20-slim). '
                f'stderr={prep.stderr.strip()[-1000:]!r}'
            )
        # Step 2: pull nodesource setup script + apt-install nodejs.
        node = await env.exec(
            [
                'bash', '-c', 'set -e; export DEBIAN_FRONTEND=noninteractive; '
                f'curl -fsSL {self._node_setup_url} | bash - && '
                'apt-get install -y --no-install-recommends nodejs'
            ],
            timeout=self._install_timeout_s,
        )
        if node.returncode != 0:
            raise RuntimeError(
                f'ClaudeCodeRunner.setup: Node.js install failed (rc={node.returncode}). '
                f'Network access from inside the sandbox is required for runtime install. '
                f'stderr={node.stderr.strip()[-1000:]!r}'
            )

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
        # Only the default-path branch (``home_override is None``) creates a
        # fresh tempdir we own; user-supplied paths and the inherit case must
        # not be deleted out from under them.
        owns_home_dir = home_dir is not None and self._home_override is None
        if home_dir is not None:
            env_vars['HOME'] = home_dir

        try:
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

            sample_id = (task.metadata or {}).get('sample_id')
            env_name = getattr(env, 'name', type(env).__name__)
            logger.info(
                f'claude-code launching: sample={sample_id} env={env_name} '
                f'model={self._model_name or "<bridge-default>"} '
                f'timeout={task.timeout}s instruction_chars={len(task.instruction)}'
            )
            result = await env.exec(
                cmd,
                timeout=task.timeout,
                env=env_vars,
            )
            logger.info(
                f'claude-code exited: sample={sample_id} rc={result.returncode} '
                f'wall={result.duration:.1f}s '
                f'stdout={len(result.stdout or "")}B stderr={len(result.stderr or "")}B '
                f'timed_out={result.timed_out}'
            )
            if result.timed_out:
                raise RunnerTimeoutError(
                    f'claude-code timed out after {task.timeout}s '
                    f'(returncode={result.returncode})'
                )
            if result.returncode != 0:
                tail_stderr = (result.stderr or '').strip()[-2000:]
                raise RuntimeError(f'claude-code exited with code {result.returncode}: {tail_stderr}')
            return AgentRunResult(
                output=result.stdout.strip(),
                metrics={
                    'wall_time': result.duration,
                    'returncode': result.returncode,
                },
            )
        finally:
            if owns_home_dir and home_dir:
                shutil.rmtree(home_dir, ignore_errors=True)

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
