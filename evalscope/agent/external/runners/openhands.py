"""Runner for the OpenHands agent framework.

Points OpenHands at the bridge via ``LLM_BASE_URL`` / ``LLM_API_KEY``
env vars.  OpenHands internally uses LiteLLM which speaks the standard
OpenAI Chat Completions API — the bridge's ``/openai/v1/chat/completions``
route handles it transparently.

The runner supports two installation modes:
1. **Pre-built image** (recommended): user builds the image from
   ``evalscope/agent/external/dockerfiles/Dockerfile.openhands`` which
   pre-installs ``openhands-ai`` into ``/opt/openhands-venv``.
2. **auto_install** (default fallback): if the probe fails, the runner
   creates the venv and ``pip install``s ``openhands-ai`` at runtime.
   Functional but slow (~2-5 min on a cold cache).
"""

import json
import shlex
from typing import Any, Dict, List, Optional

from evalscope.api.agent import AgentEnvironment
from evalscope.api.registry import register_runner
from evalscope.utils.logger import get_logger
from .base import AgentRunner, AgentRunResult, BridgeEndpoint, ExternalAgentTask, RunnerTimeoutError

logger = get_logger()

#: Path where the dedicated OpenHands venv lives (matches Dockerfile).
_OPENHANDS_VENV = '/opt/openhands-venv'
_OPENHANDS_PYTHON = f'{_OPENHANDS_VENV}/bin/python'


@register_runner('openhands')
class OpenHandsRunner(AgentRunner):
    """Drive ``openhands.core.main`` for one sample.

    Kwargs forwarded from ``ExternalAgentConfig.kwargs``:

    * ``model_name``        — model id forwarded to the bridge inside
      the LLM request body.  Runner automatically prepends ``openai/``
      for LiteLLM provider routing.
    * ``disable_tool_calls`` — when True, sets
      ``LLM_NATIVE_TOOL_CALLING=false`` so OpenHands uses text-based
      tool invocations instead of native function calling.
    * ``max_iterations``    — caps OpenHands' agent loop iterations.
    * ``auto_install``      — when True (default), installs
      ``openhands-ai`` into a venv if the probe fails.  Set False for
      pre-baked images.
    * ``install_timeout_s`` — per-step wall-clock budget (default 600s)
      for the pip install command.
    * ``extra_env``         — additional env vars passed through to the
      OpenHands process.
    """

    framework: str = 'openhands'

    #: Default wall-clock budget (seconds) for pip install.  OpenHands +
    #: all transitive deps can take 2-5 min on cold cache.
    _INSTALL_TIMEOUT_S: float = 600.0

    def __init__(
        self,
        *,
        model_name: str = '',
        disable_tool_calls: bool = False,
        max_iterations: Optional[int] = None,
        auto_install: bool = True,
        install_timeout_s: float = _INSTALL_TIMEOUT_S,
        extra_env: Optional[Dict[str, str]] = None,
        **_: Any,
    ) -> None:
        self._model_name = model_name
        self._disable_tool_calls = disable_tool_calls
        self._max_iterations = max_iterations
        self._auto_install = auto_install
        self._install_timeout_s = install_timeout_s
        self._extra_env = dict(extra_env or {})

    async def setup(self, env: AgentEnvironment) -> None:
        """Ensure OpenHands is installed inside the environment.

        Probes ``/opt/openhands-venv/bin/python -m openhands.core.main
        --version``.  If the binary is already present (pre-built image)
        we are done.  Otherwise — and only when ``auto_install=True`` —
        creates the venv and ``pip install``s ``openhands-ai``.
        """
        if await self._openhands_present(env):
            return
        if not self._auto_install:
            raise RuntimeError(
                'OpenHands not found in the agent environment and auto_install=False. '
                'Either use a pre-built image '
                '(see evalscope/agent/external/dockerfiles/Dockerfile.openhands) '
                'or pass auto_install=True.'
            )
        await self._install_openhands(env)
        if not await self._openhands_present(env):
            raise RuntimeError(
                'OpenHands install reported success but the version probe still '
                'fails.  Inspect the install logs above for the underlying cause.'
            )

    async def _openhands_present(self, env: AgentEnvironment) -> bool:
        probe = await env.exec(
            ['bash', '-c', f'{_OPENHANDS_PYTHON} -m openhands.core.main --version 2>/dev/null']
        )
        if probe.returncode == 0:
            logger.debug(f'openhands probe: {probe.stdout.strip()!r}')
            return True
        return False

    async def _install_openhands(self, env: AgentEnvironment) -> None:
        """Create venv and install openhands-ai.

        Requires a Debian/Ubuntu-based image with Python 3.10+ available.
        """
        logger.info(
            'OpenHandsRunner.setup: installing openhands-ai into '
            f'{_OPENHANDS_VENV} (this may take 2-5 minutes on cold cache).'
        )
        # Ensure system deps for pip builds (gcc etc are in python:3.11-slim).
        install = await env.exec(
            [
                'bash', '-c',
                f'set -e; '
                f'python3 -m venv {_OPENHANDS_VENV} && '
                f'{_OPENHANDS_VENV}/bin/pip install --no-cache-dir --upgrade pip && '
                f'{_OPENHANDS_VENV}/bin/pip install --no-cache-dir openhands-ai'
            ],
            timeout=self._install_timeout_s,
        )
        if install.returncode != 0:
            raise RuntimeError(
                f'OpenHandsRunner.setup: pip install openhands-ai failed '
                f'(rc={install.returncode}). '
                f'stderr={install.stderr.strip()[-1000:]!r}'
            )

    async def run(
        self,
        task: ExternalAgentTask,
        env: AgentEnvironment,
        bridge: BridgeEndpoint,
    ) -> AgentRunResult:
        # Build the LLM model name with openai/ prefix for LiteLLM routing.
        llm_model = self._model_name
        if llm_model and not llm_model.startswith('openai/'):
            llm_model = f'openai/{llm_model}'

        env_vars: Dict[str, str] = {
            # Core LLM routing — LiteLLM appends /chat/completions.
            'LLM_BASE_URL': f'{bridge.base_url}/openai/v1',
            'LLM_API_KEY': bridge.trial_token,
            'LLM_MODEL': llm_model or 'openai/default',
            # Sandbox / runtime settings.
            'RUNTIME': 'local',
            'RUN_AS_OPENHANDS': 'false',
            'SU_TO_USER': 'false',
            # Disable non-essential features to keep the eval hermetic.
            'AGENT_ENABLE_BROWSING': 'false',
            'ENABLE_BROWSER': 'false',
            'AGENT_ENABLE_PROMPT_EXTENSIONS': 'false',
            'SANDBOX_ENABLE_AUTO_LINT': 'true',
            'SKIP_DEPENDENCY_CHECK': '1',
            # Tool calling mode.
            'LLM_NATIVE_TOOL_CALLING': str(not self._disable_tool_calls).lower(),
            # Drop unknown params so LiteLLM does not error on bridge-only fields.
            'LLM_DROP_PARAMS': 'true',
        }
        if self._max_iterations is not None:
            env_vars['MAX_ITERATIONS'] = str(self._max_iterations)

        # Merge user-supplied extra env (lower priority than bridge vars).
        for k, v in self._extra_env.items():
            env_vars.setdefault(k, v)

        escaped_instruction = shlex.quote(task.instruction)
        cmd: List[str] = [
            'bash', '-c',
            f'{_OPENHANDS_PYTHON} -m openhands.core.main '
            f'--task={escaped_instruction} 2>&1'
        ]

        sample_id = (task.metadata or {}).get('sample_id')
        env_name = getattr(env, 'name', type(env).__name__)
        logger.info(
            f'openhands launching: sample={sample_id} env={env_name} '
            f'model={llm_model or "<bridge-default>"} '
            f'timeout={task.timeout}s instruction_chars={len(task.instruction)}'
        )
        result = await env.exec(cmd, timeout=task.timeout, env=env_vars)
        logger.info(
            f'openhands exited: sample={sample_id} rc={result.returncode} '
            f'wall={result.duration:.1f}s '
            f'stdout={len(result.stdout or "")}B stderr={len(result.stderr or "")}B '
            f'timed_out={result.timed_out}'
        )
        if result.timed_out:
            raise RunnerTimeoutError(
                f'openhands timed out after {task.timeout}s '
                f'(returncode={result.returncode})'
            )
        if result.returncode != 0:
            tail_stderr = (result.stderr or '').strip()[-2000:]
            tail_stdout = (result.stdout or '').strip()[-2000:]
            raise RuntimeError(
                f'openhands exited with code {result.returncode}: '
                f'{tail_stderr or tail_stdout}'
            )
        return AgentRunResult(
            output=result.stdout.strip(),
            metrics={
                'wall_time': result.duration,
                'returncode': result.returncode,
            },
        )
