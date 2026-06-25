"""Runner for Google's ``gemini`` CLI.

Points ``gemini -p`` at the bridge via ``GOOGLE_GEMINI_BASE_URL`` /
``GEMINI_API_KEY`` env vars. Gemini CLI uses the native Google AI
``generateContent`` endpoint, which the bridge translates into
EvalScope's model layer via the ``/gemini/v1beta/models/*`` route.

The Gemini CLI is a Node.js tool installed via npm
(``@google/gemini-cli``). It supports non-interactive (headless)
mode via ``-p "prompt"`` for batch execution.

Environment variables consumed by Gemini CLI:
- ``GEMINI_API_KEY`` — API key (we use the bridge token)
- ``GOOGLE_GEMINI_BASE_URL`` — base URL for model requests (bridge)

The runner defaults to ``--yolo`` (auto-approve all actions) for
batch execution.
"""

import tempfile
from typing import Any, Dict, List, Optional

from evalscope.api.agent import AgentEnvironment
from evalscope.api.registry import register_runner
from evalscope.utils.logger import get_logger
from ._node_install import ensure_node_via_apt
from .base import AgentRunner, AgentRunResult, BridgeEndpoint, ExternalAgentTask, RunnerTimeoutError

logger = get_logger()


@register_runner('gemini-cli')
class GeminiCliRunner(AgentRunner):
    """Drive ``gemini`` CLI for one sample.

    Kwargs forwarded from ``ExternalAgentConfig.kwargs``:

    * ``model_name``        — model id advertised in the bridge request.
      Gemini CLI uses this as the model name in its API calls.
    * ``extra_args``        — verbatim args appended before the prompt.
    * ``auto_install``      — when True (default), installs Node.js +
      ``@google/gemini-cli`` if ``gemini --version`` fails.
    * ``install_timeout_s`` — per-step wall-clock budget (default 300s).
    * ``home_override``     — optional ``HOME`` path. Defaults to a
      fresh per-run tempdir for isolation.
    * ``node_setup_url``    — nodesource setup script URL.
    * ``npm_package``       — npm package name to install.

    Hard-coded for batch execution:
    * ``--yolo`` — auto-approve all actions (no interactive prompts)
    """

    framework: str = 'gemini-cli'

    _INSTALL_TIMEOUT_S: float = 300.0

    def __init__(
        self,
        *,
        model_name: str = '',
        extra_args: Optional[List[str]] = None,
        auto_install: bool = True,
        install_timeout_s: float = _INSTALL_TIMEOUT_S,
        home_override: Optional[str] = None,
        node_setup_url: str = 'https://deb.nodesource.com/setup_22.x',
        npm_package: str = '@google/gemini-cli@0.43.0',
        **_: Any,
    ) -> None:
        self._model_name = model_name
        self._extra_args = list(extra_args or [])
        self._auto_install = auto_install
        self._install_timeout_s = install_timeout_s
        self._home_override = home_override
        self._node_setup_url = node_setup_url
        self._npm_package = npm_package

    # ------------------------------------------------------------------
    # setup
    # ------------------------------------------------------------------

    async def setup(self, env: AgentEnvironment) -> None:
        """Ensure Gemini CLI is installed inside the environment."""
        if await self._gemini_present(env):
            return
        if not self._auto_install:
            raise RuntimeError(
                'gemini CLI not found in the agent environment and auto_install=False. '
                'Either bake gemini-cli into the image or pass auto_install=True.'
            )
        await self._install_gemini_cli(env)
        if not await self._gemini_present(env):
            raise RuntimeError(
                'gemini CLI install reported success but `gemini --version` still fails. '
                'Inspect the install logs above for the underlying cause.'
            )

    async def _gemini_present(self, env: AgentEnvironment) -> bool:
        probe = await env.exec(['bash', '-c', 'command -v gemini && gemini --version'])
        if probe.returncode == 0:
            logger.debug(f'gemini-cli probe: {probe.stdout.strip()!r}')
            return True
        return False

    async def _install_gemini_cli(self, env: AgentEnvironment) -> None:
        """Install Node.js (when missing) and the gemini-cli package."""
        await ensure_node_via_apt(
            env,
            node_setup_url=self._node_setup_url,
            timeout_s=self._install_timeout_s,
            runner_name='GeminiCliRunner',
        )
        npm = await env.exec(
            ['bash', '-c', f'set -e; npm install -g --no-fund --no-audit {self._npm_package} >/dev/null'],
            timeout=self._install_timeout_s,
        )
        if npm.returncode != 0:
            raise RuntimeError(
                f'GeminiCliRunner.setup: `npm install -g {self._npm_package}` failed '
                f'(rc={npm.returncode}). stderr={npm.stderr.strip()[-1000:]!r}'
            )

    # ------------------------------------------------------------------
    # run
    # ------------------------------------------------------------------

    async def run(
        self,
        task: ExternalAgentTask,
        env: AgentEnvironment,
        bridge: BridgeEndpoint,
    ) -> AgentRunResult:
        env_vars: Dict[str, str] = {
            # Gemini CLI reads GOOGLE_GEMINI_BASE_URL for the API endpoint.
            # The bridge exposes Gemini routes under /gemini/ prefix.
            'GOOGLE_GEMINI_BASE_URL': f'{bridge.base_url}/gemini',
            # API key — the bridge uses this as the session token.
            # GEMINI_API_KEY is the correct env var for headless Gemini AI
            # Studio auth (not GOOGLE_API_KEY which is for Vertex AI).
            'GEMINI_API_KEY': bridge.trial_token,
            # Suppress telemetry / update checks.
            'DO_NOT_TRACK': '1',
            # Trust the workspace so gemini-cli doesn't refuse to run
            # in headless mode (requires --skip-trust or this env var).
            'GEMINI_CLI_TRUST_WORKSPACE': 'true',
        }
        home_dir = self._resolve_home()
        if home_dir is not None:
            env_vars['HOME'] = home_dir

        # Build the command.
        # gemini -p "prompt" executes non-interactively.
        cmd: List[str] = ['gemini']

        # Model selection
        if self._model_name:
            cmd.extend(['-m', self._model_name])

        # Non-interactive batch flags
        cmd.append('--yolo')  # auto-approve all actions

        # Output format — JSON for structured parsing
        cmd.extend(['--output-format', 'json'])

        # Extra user-supplied args
        cmd.extend(self._extra_args)

        # Prompt via -p flag (non-interactive mode)
        cmd.extend(['-p', task.instruction])

        sample_id = (task.metadata or {}).get('sample_id')
        env_name = getattr(env, 'name', type(env).__name__)
        logger.info(
            f'gemini-cli launching: sample={sample_id} env={env_name} '
            f'model={self._model_name or "<default>"} '
            f'timeout={task.timeout}s instruction_chars={len(task.instruction)}'
        )
        result = await env.exec(cmd, timeout=task.timeout, env=env_vars)
        logger.info(
            f'gemini-cli exited: sample={sample_id} rc={result.returncode} '
            f'wall={result.duration:.1f}s '
            f'stdout={len(result.stdout or "")}B stderr={len(result.stderr or "")}B '
            f'timed_out={result.timed_out}'
        )
        if result.timed_out:
            raise RunnerTimeoutError(
                f'gemini-cli timed out after {task.timeout}s '
                f'(returncode={result.returncode})'
            )
        if result.returncode != 0:
            tail_stderr = (result.stderr or '').strip()[-2000:]
            tail_stdout = (result.stdout or '').strip()[-2000:]
            raise RuntimeError(f'gemini-cli exited with code {result.returncode}: '
                               f'{tail_stderr or tail_stdout}')
        return AgentRunResult(
            output=result.stdout.strip(),
            metrics={
                'wall_time': result.duration,
                'returncode': result.returncode,
            },
        )

    def _resolve_home(self) -> Optional[str]:
        """Pick the HOME value for the subprocess."""
        if self._home_override == '':
            return None
        if self._home_override is not None:
            return self._home_override
        return tempfile.mkdtemp(prefix='evalscope-gemini-cli-')
