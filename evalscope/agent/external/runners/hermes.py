"""Runner for Nous Research's ``hermes`` agent CLI.

Points ``hermes chat -q`` at the bridge via ``OPENAI_BASE_URL`` /
``OPENAI_API_KEY`` env vars with ``--provider openai``.  Hermes uses the
standard OpenAI Chat Completions protocol, so the bridge's existing
``/v1/chat/completions`` route handles it without extra translation.

The Hermes Agent is a Python-based tool installed via `uv` from the
official install script.  It supports non-interactive single-query mode
via ``hermes chat -q "prompt"``.

Environment variables consumed by Hermes:
- ``OPENAI_BASE_URL``   — custom OpenAI-compatible endpoint (bridge)
- ``OPENAI_API_KEY``    — API key (we use the bridge trial token)
- ``HERMES_YOLO_MODE``  — auto-approve all tool actions (no prompts)
- ``HERMES_HOME``       — config directory override for isolation

The runner defaults to ``--yolo`` (auto-approve all actions) for
batch execution.
"""

import tempfile
from typing import Any, Dict, List, Optional

from evalscope.api.agent import AgentEnvironment
from evalscope.api.registry import register_runner
from evalscope.utils.logger import get_logger
from .base import AgentRunner, AgentRunResult, BridgeEndpoint, ExternalAgentTask, RunnerTimeoutError

logger = get_logger()


@register_runner('hermes')
class HermesRunner(AgentRunner):
    """Drive ``hermes chat -q`` for one sample.

    Kwargs forwarded from ``ExternalAgentConfig.kwargs``:

    * ``model_name``        — model id forwarded to the bridge.
    * ``extra_args``        — verbatim args appended before the prompt.
    * ``auto_install``      — when True (default), runs the official
      install script if ``hermes --version`` fails.
    * ``install_timeout_s`` — per-step wall-clock budget (default 300s).
    * ``home_override``     — optional ``HERMES_HOME`` path. Defaults to
      a fresh per-run tempdir for isolation.
    * ``provider``          — LLM provider name (default: ``openai``).
    * ``toolsets``          — comma-separated toolsets to enable
      (default: ``terminal`` for code generation tasks).
    * ``install_url``       — install script URL.

    Hard-coded for batch execution:
    * ``--yolo`` — auto-approve all actions (no interactive prompts)
    """

    framework: str = 'hermes'

    _INSTALL_TIMEOUT_S: float = 300.0

    def __init__(
        self,
        *,
        model_name: str = '',
        extra_args: Optional[List[str]] = None,
        auto_install: bool = True,
        install_timeout_s: float = _INSTALL_TIMEOUT_S,
        home_override: Optional[str] = None,
        provider: str = 'openai',
        toolsets: str = 'terminal',
        install_url: str = 'https://hermes-agent.nousresearch.com/install.sh',
        **_: Any,
    ) -> None:
        self._model_name = model_name
        self._extra_args = list(extra_args or [])
        self._auto_install = auto_install
        self._install_timeout_s = install_timeout_s
        self._home_override = home_override
        self._provider = provider
        self._toolsets = toolsets
        self._install_url = install_url

    # ------------------------------------------------------------------
    # setup
    # ------------------------------------------------------------------

    async def setup(self, env: AgentEnvironment) -> None:
        """Ensure Hermes Agent is installed inside the environment."""
        if await self._hermes_present(env):
            return
        if not self._auto_install:
            raise RuntimeError(
                'hermes CLI not found in the agent environment and auto_install=False. '
                'Either bake hermes into the image or pass auto_install=True.'
            )
        await self._install_hermes(env)
        if not await self._hermes_present(env):
            raise RuntimeError(
                'hermes install reported success but `hermes --version` still fails. '
                'Inspect the install logs above for the underlying cause.'
            )

    async def _hermes_present(self, env: AgentEnvironment) -> bool:
        probe = await env.exec(['bash', '-c', 'command -v hermes && hermes --version'])
        if probe.returncode == 0:
            logger.debug(f'hermes probe: {probe.stdout.strip()!r}')
            return True
        return False

    async def _install_hermes(self, env: AgentEnvironment) -> None:
        """Install Hermes Agent via the official install script.

        Hermes is Python-based and uses ``uv`` for dependency management.
        The install script handles Python, uv, and all dependencies.
        """
        logger.info(
            f'HermesRunner.setup: installing hermes via {self._install_url} '
            f'(one-shot per sample; use pre-built image for faster iteration).'
        )
        # Ensure curl is available
        prep = await env.exec(
            [
                'bash', '-c', 'set -e; export DEBIAN_FRONTEND=noninteractive; '
                'apt-get update -qq && '
                'apt-get install -y --no-install-recommends curl ca-certificates'
            ],
            timeout=self._install_timeout_s,
        )
        if prep.returncode != 0:
            raise RuntimeError(
                f'HermesRunner.setup: apt prerequisite install failed (rc={prep.returncode}). '
                f'This runner expects a Debian/Ubuntu-based image with network access. '
                f'stderr={prep.stderr.strip()[-1000:]!r}'
            )
        # Run the official install script
        install = await env.exec(
            ['bash', '-c', f'set -e; curl -fsSL {self._install_url} | bash'],
            timeout=self._install_timeout_s,
        )
        if install.returncode != 0:
            raise RuntimeError(
                f'HermesRunner.setup: hermes install script failed (rc={install.returncode}). '
                f'stderr={install.stderr.strip()[-1000:]!r}'
            )
        # Source shell config to get hermes on PATH
        source = await env.exec(
            ['bash', '-c', 'source ~/.bashrc 2>/dev/null; command -v hermes'],
            timeout=30,
        )
        if source.returncode != 0:
            logger.warning('hermes not on PATH after install; trying ~/.hermes/bin')

    # ------------------------------------------------------------------
    # run
    # ------------------------------------------------------------------

    async def run(
        self,
        task: ExternalAgentTask,
        env: AgentEnvironment,
        bridge: BridgeEndpoint,
    ) -> AgentRunResult:
        home_dir = self._resolve_home()
        env_vars: Dict[str, str] = {
            # Point Hermes at the bridge's OpenAI-compatible endpoint.
            # Bridge routes are at /openai/v1/chat/completions — so base_url
            # must be {bridge}/openai/v1 for Hermes to hit the right path.
            'OPENAI_BASE_URL': f'{bridge.base_url}/openai/v1',
            'OPENAI_API_KEY': bridge.trial_token,
            # Auto-approve all tool actions (no interactive prompts).
            'HERMES_YOLO_MODE': '1',
            # Suppress telemetry / update checks.
            'DO_NOT_TRACK': '1',
            # Ensure hermes binary is on PATH (symlinked to /usr/local/bin in Docker image;
            # venv at ~/.hermes/hermes-agent/.venv/bin for auto_install fallback).
            'PATH': '/usr/local/bin:/root/.hermes/hermes-agent/.venv/bin:/root/.hermes/bin:'
            '/usr/local/sbin:/usr/sbin:/usr/bin:/sbin:/bin',
        }
        if home_dir is not None:
            env_vars['HERMES_HOME'] = home_dir

        # Write a config.yaml that points Hermes at the bridge endpoint.
        # When base_url is set, Hermes ignores the provider and calls the
        # endpoint directly (using api_key or OPENAI_API_KEY for auth).
        hermes_home = home_dir or '/root/.hermes'
        config_yaml = (
            f'model:\n'
            f'  provider: custom\n'
            f'  default: {self._model_name or "default"}\n'
            f'  base_url: "{bridge.base_url}/openai/v1"\n'
            f'  api_key: "{bridge.trial_token}"\n'
            f'  context_length: 65536\n'
        )
        await env.exec(
            [
                'bash', '-c',
                f'mkdir -p {hermes_home} && cat > {hermes_home}/config.yaml << \'EOFCFG\'\n{config_yaml}EOFCFG'
            ],
            timeout=10,
        )

        # Build the command.
        cmd: List[str] = ['hermes', 'chat']

        # Model selection (also passed via config but CLI flag takes priority)
        if self._model_name:
            cmd.extend(['--model', self._model_name])

        # Toolsets
        if self._toolsets:
            cmd.extend(['--toolsets', self._toolsets])

        # Auto-approve all tool calls (belt-and-suspenders with env var)
        cmd.append('--yolo')

        # Quiet mode — suppress banner/spinner, only output final response
        cmd.append('--quiet')

        # Extra user-supplied args
        cmd.extend(self._extra_args)

        # Single-query mode (non-interactive) — must be last
        cmd.extend(['-q', task.instruction])

        sample_id = (task.metadata or {}).get('sample_id')
        env_name = getattr(env, 'name', type(env).__name__)
        logger.info(
            f'hermes launching: sample={sample_id} env={env_name} '
            f'model={self._model_name or "<default>"} '
            f'provider={self._provider} '
            f'timeout={task.timeout}s instruction_chars={len(task.instruction)}'
        )
        result = await env.exec(cmd, timeout=task.timeout, env=env_vars)
        logger.info(
            f'hermes exited: sample={sample_id} rc={result.returncode} '
            f'wall={result.duration:.1f}s '
            f'stdout={len(result.stdout or "")}B stderr={len(result.stderr or "")}B '
            f'timed_out={result.timed_out}'
        )
        if result.timed_out:
            raise RunnerTimeoutError(f'hermes timed out after {task.timeout}s '
                                     f'(returncode={result.returncode})')
        if result.returncode != 0:
            tail_stderr = (result.stderr or '').strip()[-2000:]
            tail_stdout = (result.stdout or '').strip()[-2000:]
            raise RuntimeError(
                f'hermes exited with code {result.returncode}:\n'
                f'  stderr: {tail_stderr}\n'
                f'  stdout: {tail_stdout}'
            )
        return AgentRunResult(
            output=result.stdout.strip(),
            metrics={
                'wall_time': result.duration,
                'returncode': result.returncode,
            },
        )

    def _resolve_home(self) -> Optional[str]:
        """Pick the HERMES_HOME value for the subprocess."""
        if self._home_override == '':
            return None
        if self._home_override is not None:
            return self._home_override
        return tempfile.mkdtemp(prefix='evalscope-hermes-')
