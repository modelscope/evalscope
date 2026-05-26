"""Runner for OpenAI's ``@openai/codex`` CLI.

Points ``codex exec`` at the bridge via ``-c model_providers.evalscope.*``
config overrides (codex does NOT read ``OPENAI_BASE_URL`` env vars). The
runner stays stateless: every codex invocation gets the full provider
config on the command line, avoiding any persistent ``~/.codex/config.toml``
edits.

codex v0.133+ removed Chat Completions support and always speaks the
Responses API — the bridge's ``/openai/v1/responses`` route is the only
endpoint codex can actually hit.

The prompt is passed as the trailing positional argument (codex ``exec
"<prompt>"``) so this runner works in both ``LocalAgentEnvironment``
(stdin is fully supported) and ``EnclaveAgentEnvironment`` (whose
``exec`` does not forward ``input=`` to the underlying shell executor).
"""

import tempfile
from typing import Any, Dict, List, Optional

from evalscope.api.agent import AgentEnvironment
from evalscope.api.registry import register_runner
from evalscope.utils.logger import get_logger
from .base import AgentRunner, AgentRunResult, BridgeEndpoint, ExternalAgentTask, RunnerTimeoutError

logger = get_logger()

#: codex writes its final assistant message here; the runner reads it back.
#: Hard-coded — implementation detail, never exposed as a kwarg. If a future
#: sandbox forbids writes to /tmp, switch this constant in lockstep with the
#: caller side and add a kwarg.
_CODEX_OUTPUT_FILE = '/tmp/evalscope-codex-last.txt'

#: codex sandbox mode used by the runner. ``workspace-write`` lets codex
#: edit files under its CWD + run shell commands while still blocking writes
#: outside the workspace — safe default for both Local and Enclave runs.
#: ``danger-full-access`` would lift all in-process limits, which is unsafe
#: for ``LocalAgentEnvironment`` (codex would have full host access). Bench
#: cases that genuinely need full access can be re-introduced behind an
#: explicit kwarg later; PR2 SWE-bench Pro verified ``workspace-write`` is
#: sufficient for ``apply_patch`` + ``exec_command``.
_CODEX_SANDBOX_MODE = 'workspace-write'


@register_runner('codex')
class CodexRunner(AgentRunner):
    """Drive ``codex exec`` for one sample.

    Kwargs forwarded from ``ExternalAgentConfig.kwargs`` (all optional —
    pass an empty ``kwargs={}`` for the default behaviour):

    * ``model_name``  — model id the bridge dials. Defaults to the model
      configured on :class:`TaskConfig` (the adapter auto-injects it).
      Override only when you want codex to advertise a different model
      string than evalscope-side ``Model``.
    * ``extra_args``  — verbatim args appended before the prompt
      positional.
    * ``extra_config`` — additional ``-c key=value`` codex overrides
      (already-quoted TOML values).
    * ``home_override`` — optional ``HOME`` path. Defaults to a fresh
      per-run tempdir so codex cannot reuse a logged-in token from the
      host keychain (mirrors :class:`ClaudeCodeRunner`'s default).
    * ``auto_install`` — when True (default), apt+nodesource+npm installs
      ``@openai/codex`` if ``codex --version`` fails on probe. Set False
      for pre-baked images.
    * ``install_timeout_s`` — per-step wall-clock budget (default 600s)
      for the apt / nodesource / npm install commands. Bump on slow
      networks or when the npm registry / GitHub CDN is rate-limiting.

    Hard-coded for safety / simplicity (not exposed as kwargs):

    * Sandbox mode = ``workspace-write`` (safe for both Local and
      Enclave; sufficient for SWE-bench Pro per PR2 verification)
    * ``--dangerously-bypass-approvals-and-sandbox`` always set
      (evalscope is batch-only; interactive prompts have no operator)
    * Final answer file = ``/tmp/evalscope-codex-last.txt``
    """

    framework: str = 'codex'

    #: Default wall-clock budget (seconds) per install step. Overridable per-instance
    #: via the ``install_timeout_s`` kwarg.
    _INSTALL_TIMEOUT_S: float = 600.0

    def __init__(
        self,
        *,
        model_name: str = '',
        extra_args: Optional[List[str]] = None,
        extra_config: Optional[Dict[str, str]] = None,
        home_override: Optional[str] = None,
        auto_install: bool = True,
        install_timeout_s: float = _INSTALL_TIMEOUT_S,
        node_setup_url: str = 'https://deb.nodesource.com/setup_20.x',
        npm_package: str = '@openai/codex',
        **_: Any,
    ) -> None:
        self._model_name = model_name
        self._extra_args = list(extra_args or [])
        self._extra_config = dict(extra_config or {})
        self._home_override = home_override
        self._auto_install = auto_install
        self._install_timeout_s = install_timeout_s
        self._node_setup_url = node_setup_url
        self._npm_package = npm_package

    # ------------------------------------------------------------------
    # setup
    # ------------------------------------------------------------------

    async def setup(self, env: AgentEnvironment) -> None:
        if await self._codex_present(env):
            return
        if not self._auto_install:
            raise RuntimeError(
                'codex CLI not found in the agent environment and auto_install=False. '
                'Either bake codex into the image or pass auto_install=True.'
            )
        await self._install_codex_cli(env)
        if not await self._codex_present(env):
            raise RuntimeError(
                'codex CLI install reported success but `codex --version` still fails. '
                'Inspect the install logs above for the underlying cause.'
            )

    async def _codex_present(self, env: AgentEnvironment) -> bool:
        probe = await env.exec(['bash', '-c', 'command -v codex && codex --version'])
        if probe.returncode == 0:
            logger.debug(f'codex probe: {probe.stdout.strip()!r}')
            return True
        return False

    async def _install_codex_cli(self, env: AgentEnvironment) -> None:
        """Install Node.js (when missing) and the ``@openai/codex`` package.

        Mirrors :meth:`ClaudeCodeRunner._install_claude_code` structure —
        kept as a separate copy on purpose (this is only the second
        runner to need this routine; if a third appears, extract to
        :mod:`base`).
        """
        if not await self._node_present(env):
            await self._install_node_via_apt(env)
        npm = await env.exec(
            ['bash', '-c', f'set -e; npm install -g --no-fund --no-audit {self._npm_package} >/dev/null'],
            timeout=self._install_timeout_s,
        )
        if npm.returncode != 0:
            raise RuntimeError(
                f'CodexRunner.setup: `npm install -g {self._npm_package}` failed '
                f'(rc={npm.returncode}). stderr={npm.stderr.strip()[-1000:]!r}'
            )

    async def _node_present(self, env: AgentEnvironment) -> bool:
        probe = await env.exec(['bash', '-c', 'command -v node && command -v npm'])
        return probe.returncode == 0

    async def _install_node_via_apt(self, env: AgentEnvironment) -> None:
        logger.info(
            f'CodexRunner.setup: installing Node.js via {self._node_setup_url} '
            f'(one-shot per sample; npm cache volume optimisation planned).'
        )
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
                f'CodexRunner.setup: apt prerequisite install failed (rc={prep.returncode}). '
                f'This runner expects a Debian/Ubuntu-based image with network access, or a '
                f'base image where Node.js is already installed. '
                f'stderr={prep.stderr.strip()[-1000:]!r}'
            )
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
                f'CodexRunner.setup: Node.js install failed (rc={node.returncode}). '
                f'stderr={node.stderr.strip()[-1000:]!r}'
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
            'EVALSCOPE_BRIDGE_TOKEN': bridge.trial_token,
            # codex's telemetry / auto-update probes — same suppression
            # pattern as ClaudeCodeRunner. Belt-and-suspenders: workspace-write
            # sandbox already cuts most egress.
            'IS_SANDBOX': '1',
        }
        home_dir = self._resolve_home()
        if home_dir is not None:
            env_vars['HOME'] = home_dir

        # Build -c overrides. Order: builtin (provider config) → user extras.
        # codex parses these as TOML literals, so string values need shell-
        # escaped double quotes; the list-form ``cmd`` carries them as a
        # single argv entry, which env.exec quotes for the shell.
        config_pairs: List[str] = [
            f'model_provider="evalscope"',
            f'model_providers.evalscope.name="EvalScope Bridge"',
            f'model_providers.evalscope.base_url="{bridge.base_url}/openai/v1"',
            f'model_providers.evalscope.env_key="EVALSCOPE_BRIDGE_TOKEN"',
            f'model_providers.evalscope.wire_api="responses"',
        ]
        if self._model_name:
            config_pairs.append(f'model="{self._model_name}"')
        for k, v in self._extra_config.items():
            config_pairs.append(f'{k}={v}')

        cmd: List[str] = ['codex', 'exec']
        for pair in config_pairs:
            cmd.extend(['-c', pair])
        cmd.extend(['--sandbox', _CODEX_SANDBOX_MODE])
        # Always non-interactive: evalscope is a batch harness, there is no
        # operator to answer codex's permission prompts.
        cmd.append('--dangerously-bypass-approvals-and-sandbox')
        cmd.extend(['--output-last-message', _CODEX_OUTPUT_FILE])
        cmd.extend(self._extra_args)
        # Positional prompt avoids the Enclave stdin gap (ms_enclave's
        # shell_executor does not pipe stdin to the child process).
        cmd.append(task.instruction)

        sample_id = (task.metadata or {}).get('sample_id')
        env_name = getattr(env, 'name', type(env).__name__)
        logger.info(
            f'codex launching: sample={sample_id} env={env_name} '
            f'model={self._model_name or "<bridge-default>"} '
            f'timeout={task.timeout}s instruction_chars={len(task.instruction)}'
        )
        result = await env.exec(cmd, timeout=task.timeout, env=env_vars)
        logger.info(
            f'codex exited: sample={sample_id} rc={result.returncode} '
            f'wall={result.duration:.1f}s '
            f'stdout={len(result.stdout or "")}B stderr={len(result.stderr or "")}B '
            f'timed_out={result.timed_out}'
        )
        if result.timed_out:
            raise RunnerTimeoutError(f'codex timed out after {task.timeout}s (returncode={result.returncode})')
        if result.returncode != 0:
            tail_stderr = (result.stderr or '').strip()[-2000:]
            raise RuntimeError(f'codex exited with code {result.returncode}: {tail_stderr}')

        # codex writes the final assistant message to --output-last-message.
        # We read via a separate exec because env.exec doesn't expose
        # arbitrary file reads. ``|| true`` so a missing file (codex
        # never produced a final message) yields empty rather than a
        # spurious non-zero exit.
        cat = await env.exec(['bash', '-c', f'cat {_CODEX_OUTPUT_FILE} 2>/dev/null || true'])
        output = cat.stdout.strip()
        if not output:
            logger.warning(
                f'codex: --output-last-message file {_CODEX_OUTPUT_FILE!r} '
                f'empty or unreadable; final answer extraction may fail downstream'
            )
        return AgentRunResult(
            output=output,
            metrics={
                'wall_time': result.duration,
                'returncode': result.returncode,
            },
        )

    def _resolve_home(self) -> Optional[str]:
        """``None`` means inherit; empty-string override also inherits; any
        other string is used verbatim. Default (None field value) gives a
        fresh tempdir so codex can't reuse a host keychain token."""
        if self._home_override == '':
            return None
        if self._home_override is not None:
            return self._home_override
        return tempfile.mkdtemp(prefix='evalscope-codex-')
