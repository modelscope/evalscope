"""Runner for DeepSeek Harness's ``dsh`` headless profile."""

import json
import shlex
import shutil
import tempfile
from typing import Any, Dict, List, Optional

from evalscope.api.agent import AgentEnvironment
from evalscope.api.registry import register_runner
from evalscope.utils.logger import get_logger

from .base import AgentRunner, AgentRunResult, BridgeEndpoint, ExternalAgentTask, RunnerTimeoutError
from .install_helper import ensure_node_via_apt, install_task_skills

logger = get_logger()


@register_runner('deepseek-harness')
class DeepSeekHarnessRunner(AgentRunner):
    """Drive ``dsh --profile headless`` for one sample."""

    framework: str = 'deepseek-harness'
    _INSTALL_TIMEOUT_S: float = 300.0
    _DEFAULT_DSH_VERSION: str = '0.1.5-rc.2'

    def __init__(
        self,
        *,
        model_name: str = '',
        extra_args: Optional[List[str]] = None,
        auto_install: bool = True,
        dsh_version: str = _DEFAULT_DSH_VERSION,
        permission_mode: str = 'danger-full-access',
        install_timeout_s: float = _INSTALL_TIMEOUT_S,
        home_override: Optional[str] = None,
        node_setup_url: str = 'https://deb.nodesource.com/setup_22.x',
        **_: Any,
    ) -> None:
        self._model_name = model_name
        self._extra_args = list(extra_args or [])
        self._auto_install = auto_install
        self._dsh_version = dsh_version
        self._permission_mode = permission_mode
        self._install_timeout_s = install_timeout_s
        self._home_override = home_override
        self._node_setup_url = node_setup_url

    async def setup(self, env: AgentEnvironment) -> None:
        """Ensure DeepSeek Harness is installed inside the environment."""
        if await self._dsh_present(env):
            return
        if not self._auto_install:
            raise RuntimeError(
                'dsh CLI not found in the agent environment and auto_install=False. '
                'Either use evalscope/agent/external/dockerfiles/Dockerfile.deepseek-harness '
                'or pass auto_install=True.'
            )
        await self._install_dsh(env)
        if not await self._dsh_present(env):
            raise RuntimeError(
                'dsh install reported success but `dsh --help` still fails. '
                'Inspect the install logs above for the underlying cause.'
            )

    async def _dsh_present(self, env: AgentEnvironment) -> bool:
        probe = await env.exec(['bash', '-c', 'command -v dsh && dsh --help'])
        if probe.returncode == 0:
            logger.debug(f'deepseek-harness probe: {probe.stdout.strip()!r}')
            return True
        return False

    async def _install_dsh(self, env: AgentEnvironment) -> None:
        """Install the pinned DSH npm package."""
        await ensure_node_via_apt(
            env,
            node_setup_url=self._node_setup_url,
            timeout_s=self._install_timeout_s,
            runner_name='DeepSeekHarnessRunner',
        )
        package = f'@deepseek-ai/dsh@{self._dsh_version}'
        install = await env.exec(
            ['bash', '-c', f'set -e; npm install -g --no-fund --no-audit {shlex.quote(package)} >/dev/null'],
            timeout=self._install_timeout_s,
        )
        if install.returncode != 0:
            raise RuntimeError(
                f'DeepSeekHarnessRunner.setup: `npm install -g {package}` failed '
                f'(rc={install.returncode}). stderr={install.stderr.strip()[-1000:]!r}'
            )

    async def run(
        self,
        task: ExternalAgentTask,
        env: AgentEnvironment,
        bridge: BridgeEndpoint,
    ) -> AgentRunResult:
        home_dir = self._resolve_home()
        owns_home_dir = home_dir is not None and self._home_override is None
        env_vars: Dict[str, str] = {
            'OPENAI_API_KEY': bridge.trial_token,
            'DSH_PERMISSION_MODE': self._permission_mode,
        }
        if home_dir is not None:
            env_vars['DSH_HOME'] = home_dir

        try:
            await install_task_skills(
                env,
                task,
                home_dir=home_dir,
                native_install_paths=[f'{home_dir}/skills'] if home_dir else [],
                runner_name='DeepSeekHarnessRunner',
            )
            await self._write_settings(env, env_vars, bridge)
            cmd = ['dsh', '--profile', 'headless']
            cmd.extend(self._extra_args)
            cmd.append(task.instruction)

            sample_id = (task.metadata or {}).get('sample_id')
            env_name = getattr(env, 'name', type(env).__name__)
            logger.info(
                f'deepseek-harness launching: sample={sample_id} env={env_name} '
                f'model={self._model_name or "default"} '
                f'timeout={task.timeout}s instruction_chars={len(task.instruction)}'
            )
            result = await env.exec(cmd, timeout=task.timeout, env=env_vars)
            logger.info(
                f'deepseek-harness exited: sample={sample_id} rc={result.returncode} '
                f'wall={result.duration:.1f}s '
                f'stdout={len(result.stdout or "")}B stderr={len(result.stderr or "")}B '
                f'timed_out={result.timed_out}'
            )
            if result.timed_out:
                raise RunnerTimeoutError(
                    f'deepseek-harness timed out after {task.timeout}s (returncode={result.returncode})'
                )
            if result.returncode != 0:
                tail_stderr = (result.stderr or '').strip()[-2000:]
                tail_stdout = (result.stdout or '').strip()[-2000:]
                raise RuntimeError(
                    f'deepseek-harness exited with code {result.returncode}: {tail_stderr or tail_stdout}'
                )
            return AgentRunResult(
                output=result.stdout.strip(),
                metrics={'wall_time': result.duration, 'returncode': result.returncode},
            )
        finally:
            if owns_home_dir and home_dir:
                shutil.rmtree(home_dir, ignore_errors=True)

    async def _write_settings(
        self,
        env: AgentEnvironment,
        env_vars: Dict[str, str],
        bridge: BridgeEndpoint,
    ) -> None:
        settings = {
            'llm-pi-ai': {
                'providers': {
                    'evalscope-bridge': {
                        'apiKeyEnv': 'OPENAI_API_KEY',
                        'api': 'openai-completions',
                        'baseURL': f'{bridge.base_url}/openai/v1',
                        'compat': {
                            'supportsDeveloperRole': False,
                            'maxTokensField': 'max_tokens',
                        },
                        'models': [{'id': self._model_name or 'default'}],
                    }
                }
            },
            'agent-default-model': {
                'provider': 'evalscope-bridge',
                'model': self._model_name or 'default',
            },
        }
        content = shlex.quote(json.dumps(settings))
        write = await env.exec(
            [
                'bash',
                '-c',
                f'settings_dir="${{DSH_HOME:-$HOME/.dsh}}"; mkdir -p "$settings_dir"; '
                f'printf %s {content} > "$settings_dir/settings.yaml"',
            ],
            timeout=10,
            env=env_vars,
        )
        if write.returncode != 0:
            raise RuntimeError(
                f'DeepSeekHarnessRunner: failed to write DSH settings (rc={write.returncode}). '
                f'stderr={write.stderr.strip()[-1000:]!r}'
            )

    def _resolve_home(self) -> Optional[str]:
        """Pick the DSH_HOME value for the subprocess."""
        if self._home_override == '':
            return None
        if self._home_override is not None:
            return self._home_override
        return tempfile.mkdtemp(prefix='evalscope-dsh-')
