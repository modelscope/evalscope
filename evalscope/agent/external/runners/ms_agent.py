"""Runner for ModelScope's ``ms-agent`` framework.

Writes a one-shot Python entrypoint into the sandbox that instantiates
``ms_agent.LLMAgent`` with the bridge URL as its OpenAI base URL. The
agent drives its own multi-turn loop internally — tool management,
function-calling, memory, etc. are all handled by ms-agent; the bridge
only proxies LLM traffic and records the trace.

The entrypoint prints the last assistant message to stdout so the runner
can recover the final prediction the same way ``ClaudeCodeRunner`` does.
"""

import asyncio
import base64
import copy
import shlex
import textwrap
import yaml
from typing import Any, Dict, List, Optional

from evalscope.api.agent import AgentEnvironment
from evalscope.api.registry import register_runner
from evalscope.utils.logger import get_logger
from .base import AgentRunner, AgentRunResult, BridgeEndpoint, ExternalAgentTask, RunnerTimeoutError

logger = get_logger()

_DEFAULT_SYSTEM = 'You are a helpful assistant.'

_ENTRYPOINT_TEMPLATE = textwrap.dedent(
    """\
    import asyncio
    import sys

    from omegaconf import OmegaConf

    config = OmegaConf.load({config_path!r})
    instruction = sys.argv[1]

    from ms_agent import LLMAgent
    agent = LLMAgent(config=config)
    messages = asyncio.run(agent.run(instruction))

    final = ''
    for msg in reversed(messages):
        if msg.role == 'assistant' and msg.content:
            final = msg.content
            break
    sys.stdout.write(final)
"""
)

_CONFIG_PATH = '/tmp/evalscope_ms_agent_config.yaml'
_ENTRYPOINT_PATH = '/tmp/evalscope_ms_agent_entry.py'
_ENTRYPOINT = _ENTRYPOINT_TEMPLATE.format(config_path=_CONFIG_PATH)


@register_runner('ms-agent')
class MSAgentRunner(AgentRunner):
    """Drive ``ms-agent`` LLMAgent for one sample.

    Kwargs forwarded from ``ExternalAgentConfig.kwargs``:

    * ``model_name``      — model id the bridge dials.
    * ``config_file``     — local path to a ms-agent ``agent.yaml``.
      When provided the runner copies it into the sandbox and only
      overrides LLM connection settings. When absent the runner builds
      a minimal config from kwargs.
    * ``max_chat_round``  — maximum agent loop iterations (default 30).
    * ``system_prompt``   — system prompt injected into the config.
    * ``tools``           — dict of ms-agent tools config (e.g.
      ``{'code_executor': {'implementation': 'python_env'}}``).
      Merged into the ``tools`` section of the YAML config.
    * ``auto_install``    — when True (default), ``pip install ms-agent``
      if the package is not already available in the sandbox.
    * ``install_timeout_s`` — wall-clock budget for pip install
      (default 300s).
    * ``extra_pip_args``  — additional arguments appended to the pip
      install command (e.g. ``['--index-url', '...']``).
    """

    framework: str = 'ms-agent'

    _INSTALL_TIMEOUT_S: float = 300.0

    def __init__(
        self,
        *,
        model_name: str = '',
        config_file: Optional[str] = None,
        max_chat_round: int = 30,
        system_prompt: Optional[str] = None,
        tools: Optional[Dict[str, Any]] = None,
        auto_install: bool = True,
        install_timeout_s: float = _INSTALL_TIMEOUT_S,
        extra_pip_args: Optional[List[str]] = None,
        **_: Any,
    ) -> None:
        self._model_name = model_name
        self._config_file = config_file
        self._max_chat_round = max_chat_round
        self._system_prompt = system_prompt
        self._tools = tools
        self._auto_install = auto_install
        self._install_timeout_s = install_timeout_s
        self._extra_pip_args = list(extra_pip_args or [])

        # Cache user config to avoid re-reading from disk on every sample.
        self._base_cfg: Optional[Dict[str, Any]] = None
        if self._config_file is not None:
            try:
                with open(self._config_file, 'r') as f:
                    self._base_cfg = yaml.safe_load(f) or {}
            except (FileNotFoundError, yaml.YAMLError) as exc:
                raise ValueError(f'Failed to load ms-agent config file {self._config_file!r}: {exc}') from exc

    # ------------------------------------------------------------------
    # setup
    # ------------------------------------------------------------------

    async def setup(self, env: AgentEnvironment) -> None:
        if await self._ms_agent_present(env):
            return
        if not self._auto_install:
            raise RuntimeError(
                'ms-agent not found in the agent environment and auto_install=False. '
                'Either pre-install ms-agent in the image or pass auto_install=True.'
            )
        await self._install_ms_agent(env)
        if not await self._ms_agent_present(env):
            raise RuntimeError(
                'ms-agent install reported success but `import ms_agent` still fails. '
                'Inspect the install logs above for the underlying cause.'
            )

    async def _ms_agent_present(self, env: AgentEnvironment) -> bool:
        probe = await env.exec(['python', '-c', 'import ms_agent; print(ms_agent.__file__)'])
        if probe.returncode == 0:
            logger.debug(f'ms-agent probe: {probe.stdout.strip()!r}')
            return True
        return False

    async def _install_ms_agent(self, env: AgentEnvironment) -> None:
        """Install ms-agent inside the sandbox via pip."""
        logger.info('MSAgentRunner.setup: installing ms-agent via pip.')
        pip_cmd = ['pip', 'install', '--break-system-packages', 'ms-agent']
        pip_cmd.extend(self._extra_pip_args)
        pip_cmd_str = ' '.join(shlex.quote(a) for a in pip_cmd)
        result = await env.exec(
            ['bash', '-c', f'set -e; {pip_cmd_str} >/dev/null'],
            timeout=self._install_timeout_s,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f'MSAgentRunner.setup: `pip install ms-agent` failed '
                f'(rc={result.returncode}). stderr={result.stderr.strip()[-1000:]!r}'
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
        config_yaml = self._build_config_yaml(bridge)

        await asyncio.gather(
            self._write_file(env, _CONFIG_PATH, config_yaml),
            self._write_file(env, _ENTRYPOINT_PATH, _ENTRYPOINT),
        )

        sample_id = (task.metadata or {}).get('sample_id')
        env_name = getattr(env, 'name', type(env).__name__)
        logger.info(
            f'ms-agent launching: sample={sample_id} env={env_name} '
            f'model={self._model_name or "<bridge-default>"} '
            f'max_chat_round={self._max_chat_round} '
            f'timeout={task.timeout}s instruction_chars={len(task.instruction)}'
        )

        result = await env.exec(
            ['python', _ENTRYPOINT_PATH, task.instruction],
            timeout=task.timeout,
        )

        logger.info(
            f'ms-agent exited: sample={sample_id} rc={result.returncode} '
            f'wall={result.duration:.1f}s '
            f'stdout={len(result.stdout or "")}B stderr={len(result.stderr or "")}B '
            f'timed_out={result.timed_out}'
        )

        if result.timed_out:
            raise RunnerTimeoutError(f'ms-agent timed out after {task.timeout}s '
                                     f'(returncode={result.returncode})')
        if result.returncode != 0:
            tail_stderr = (result.stderr or '').strip()[-2000:]
            raise RuntimeError(f'ms-agent exited with code {result.returncode}: {tail_stderr}')

        return AgentRunResult(
            output=result.stdout.strip(),
            metrics={
                'wall_time': result.duration,
                'returncode': result.returncode,
            },
        )

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------

    def _build_config_yaml(self, bridge: BridgeEndpoint) -> str:
        """Build the ms-agent YAML config as a string.

        When ``config_file`` is set, reads the user-supplied file and
        overlays the bridge connection settings. Otherwise builds a
        minimal config from kwargs.
        """
        cfg = copy.deepcopy(self._base_cfg) if self._base_cfg is not None else {}

        if not isinstance(cfg.get('llm'), dict):
            cfg['llm'] = {}
        if not isinstance(cfg.get('generation_config'), dict):
            cfg['generation_config'] = {}

        cfg['llm']['service'] = 'openai'
        cfg['llm']['openai_base_url'] = f'{bridge.base_url}/openai/v1'
        cfg['llm']['openai_api_key'] = bridge.trial_token
        if self._model_name:
            cfg['llm']['model'] = self._model_name

        cfg['generation_config']['stream'] = False

        # Merge tools config from kwargs (if provided).
        if self._tools:
            existing_tools = cfg.get('tools', {})
            if not isinstance(existing_tools, dict):
                existing_tools = {}
            merged = copy.deepcopy(existing_tools)
            merged.update(self._tools)
            cfg['tools'] = merged

        if self._config_file is None:
            cfg['max_chat_round'] = self._max_chat_round
            cfg.setdefault('prompt', {})
            cfg['prompt']['system'] = self._system_prompt or _DEFAULT_SYSTEM
            cfg.pop('callbacks', None)

        return yaml.dump(cfg, default_flow_style=False, allow_unicode=True)

    @staticmethod
    async def _write_file(env: AgentEnvironment, path: str, content: str) -> None:
        """Write a text file inside the sandbox via base64-encoded echo.

        The ``EnclaveAgentEnvironment.exec`` does not forward stdin to the
        underlying shell executor, so we encode the content as base64 and
        decode it inside the container.
        """
        encoded = base64.b64encode(content.encode('utf-8')).decode('ascii')
        result = await env.exec(['bash', '-c', f'echo {encoded} | base64 -d > "$1"', '_', path], )
        if result.returncode != 0:
            raise RuntimeError(
                f'Failed to write {path} inside sandbox: '
                f'rc={result.returncode} stderr={result.stderr.strip()[-500:]!r}'
            )
