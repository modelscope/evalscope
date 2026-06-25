"""Runner for the OpenCode agent framework.

Points OpenCode at the bridge via ``OPENAI_BASE_URL`` / ``OPENAI_API_KEY``
env vars.  OpenCode's ``openai`` provider speaks the **OpenAI Responses
API** (``/openai/v1/responses``), not Chat Completions.  The bridge's
Responses route translates the request to ``ChatMessage[]`` and calls
``Model.generate_async`` transparently.

The runner supports two installation modes:
1. **Pre-built image** (recommended): user builds the image from
   ``evalscope/agent/external/dockerfiles/Dockerfile.opencode`` which
   pre-installs ``opencode-ai`` globally.
2. **auto_install** (default fallback): if the probe fails, the runner
   installs Node.js (via nodesource) + ``opencode-ai`` (via npm).
   Functional but adds ~30-60s on first invocation.

OpenCode requires the model name in ``provider/model`` format.  This
runner forces the ``openai`` provider and registers the model via the
``~/.config/opencode/opencode.json`` configuration file so OpenCode
recognises arbitrary model names beyond its built-in list.
"""

import json
import shlex
import shutil
import tempfile
from typing import Any, Dict, List, Optional

from evalscope.api.agent import AgentEnvironment
from evalscope.api.registry import register_runner
from evalscope.utils.logger import get_logger
from ._node_install import ensure_node_via_apt
from .base import AgentRunner, AgentRunResult, BridgeEndpoint, ExternalAgentTask, RunnerTimeoutError

logger = get_logger()


@register_runner('opencode')
class OpenCodeRunner(AgentRunner):
    """Drive ``opencode run`` for one sample.

    Kwargs forwarded from ``ExternalAgentConfig.kwargs``:

    * ``model_name``        — model id forwarded to the bridge inside
      the LLM request body.  Runner automatically prepends ``openai/``
      for provider routing.
    * ``extra_args``        — verbatim args appended before the prompt.
    * ``auto_install``      — when True (default), installs Node.js +
      ``opencode-ai`` if ``opencode --version`` fails on probe.  Set
      False for pre-baked images.
    * ``install_timeout_s`` — per-step wall-clock budget (default 300s)
      for the apt / nodesource / npm install commands.
    * ``home_override``     — optional ``HOME`` path.  Defaults to a
      fresh per-run tempdir so opencode cannot reuse a host token.
    * ``node_setup_url``    — nodesource setup script URL.
    * ``npm_package``       — npm package name to install.

    Hard-coded:
    * ``--dangerously-skip-permissions`` — evalscope is batch-only.
    * ``--format=json`` — structured output for trajectory parsing.
    * ``OPENCODE_FAKE_VCS=git`` — fake VCS so opencode doesn't complain.
    """

    framework: str = 'opencode'

    #: Default wall-clock budget (seconds) per install step.
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
        npm_package: str = 'opencode-ai',
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
        """Ensure OpenCode is installed inside the environment.

        Probes ``opencode --version``.  If the binary is already present
        (pre-built image) we are done.  Otherwise — and only when
        ``auto_install=True`` — installs Node.js + opencode-ai via npm.
        """
        if await self._opencode_present(env):
            return
        if not self._auto_install:
            raise RuntimeError(
                'opencode CLI not found in the agent environment and auto_install=False. '
                'Either use a pre-built image '
                '(see evalscope/agent/external/dockerfiles/Dockerfile.opencode) '
                'or pass auto_install=True.'
            )
        await self._install_opencode(env)
        if not await self._opencode_present(env):
            raise RuntimeError(
                'opencode CLI install reported success but `opencode --version` '
                'still fails.  Inspect the install logs above for the underlying cause.'
            )

    async def _opencode_present(self, env: AgentEnvironment) -> bool:
        probe = await env.exec(['bash', '-c', 'command -v opencode && opencode --version'])
        if probe.returncode == 0:
            logger.debug(f'opencode probe: {probe.stdout.strip()!r}')
            return True
        return False

    async def _install_opencode(self, env: AgentEnvironment) -> None:
        """Install Node.js (when missing) and opencode-ai."""
        await ensure_node_via_apt(
            env,
            node_setup_url=self._node_setup_url,
            timeout_s=self._install_timeout_s,
            runner_name='OpenCodeRunner',
        )
        npm = await env.exec(
            ['bash', '-c', f'set -e; npm install -g --no-fund --no-audit {shlex.quote(self._npm_package)} >/dev/null'],
            timeout=self._install_timeout_s,
        )
        if npm.returncode != 0:
            raise RuntimeError(
                f'OpenCodeRunner.setup: `npm install -g {self._npm_package}` failed '
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
        # Build model name in provider/model format.
        model_id = self._model_name or 'default'
        model_name = model_id if model_id.startswith('openai/') else f'openai/{model_id}'

        env_vars: Dict[str, str] = {
            'OPENAI_BASE_URL': f'{bridge.base_url}/openai/v1',
            'OPENAI_API_KEY': bridge.trial_token,
            # Fake VCS so opencode doesn't complain about missing git repo.
            'OPENCODE_FAKE_VCS': 'git',
        }
        home_dir = self._resolve_home()
        owns_home_dir = home_dir is not None and self._home_override is None
        if home_dir is not None:
            env_vars['HOME'] = home_dir

        try:
            # Write opencode.json config to register the model + baseURL.
            config = {
                'provider': {
                    'openai': {
                        'models': {
                            model_id: {}
                        },
                        'options': {
                            'baseURL': f'{bridge.base_url}/openai/v1'
                        },
                    }
                }
            }
            config_json = json.dumps(config, indent=2)
            escaped_config = shlex.quote(config_json)
            config_cmd = (
                'mkdir -p ~/.config/opencode && '
                f'echo {escaped_config} > ~/.config/opencode/opencode.json'
            )

            # Write config first.
            cfg_result = await env.exec(
                ['bash', '-c', config_cmd],
                env=env_vars,
            )
            if cfg_result.returncode != 0:
                logger.warning(
                    f'opencode: config write failed (rc={cfg_result.returncode}); '
                    f'proceeding anyway — opencode may not recognise the model'
                )

            # Build the command as a plain argv list.  The environment
            # handles shell wrapping (enclave) or direct exec (local);
            # keeping stderr separate improves diagnostics.
            cmd: List[str] = [
                'opencode',
                f'--model={model_name}',
                'run',
                '--format=json',
                '--thinking',
                '--dangerously-skip-permissions',
                # Provide an explicit title so opencode skips the LLM title-
                # generation call it normally makes as the very first request.
                # Without this, step-0 burns ~600 tokens on a "You are a title
                # generator" round-trip and pollutes the bridge transcript with
                # the title system prompt instead of the actual task context.
                '--title=sample-{}'.format((task.metadata or {}).get('sample_id', 'eval')),
            ]
            cmd.extend(self._extra_args)
            cmd.append('--')
            cmd.append(task.instruction)

            sample_id = (task.metadata or {}).get('sample_id')
            env_name = getattr(env, 'name', type(env).__name__)
            logger.info(
                f'opencode launching: sample={sample_id} env={env_name} '
                f'model={model_name} '
                f'timeout={task.timeout}s instruction_chars={len(task.instruction)}'
            )
            result = await env.exec(cmd, timeout=task.timeout, env=env_vars)
            logger.info(
                f'opencode exited: sample={sample_id} rc={result.returncode} '
                f'wall={result.duration:.1f}s '
                f'stdout={len(result.stdout or "")}B stderr={len(result.stderr or "")}B '
                f'timed_out={result.timed_out}'
            )
            if result.timed_out:
                raise RunnerTimeoutError(
                    f'opencode timed out after {task.timeout}s '
                    f'(returncode={result.returncode})'
                )
            if result.returncode != 0:
                tail_stderr = (result.stderr or '').strip()[-2000:]
                tail_stdout = (result.stdout or '').strip()[-2000:]
                raise RuntimeError(f'opencode exited with code {result.returncode}: '
                                   f'{tail_stderr or tail_stdout}')
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
        """Pick the HOME value for the subprocess.

        None means inherit.  Empty-string override also inherits.  Any
        other string is used verbatim.  Default (None field value)
        creates a fresh tempdir so opencode can't reuse a host token.
        """
        if self._home_override == '':
            return None
        if self._home_override is not None:
            return self._home_override
        return tempfile.mkdtemp(prefix='evalscope-opencode-')
