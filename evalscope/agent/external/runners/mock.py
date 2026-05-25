"""Black-box mock agent used for end-to-end bridge smoke tests.

The runner spawns a subprocess that speaks the Anthropic Messages API
back to the bridge (single turn, no tools).  It is intentionally
self-contained — pure ``urllib`` so it inherits zero deps from
EvalScope — so that running it through ``LocalAgentEnvironment`` is a
real walking-skeleton validation that the bridge is reachable and
parses Anthropic traffic correctly.
"""

import json
import sys
from typing import Any, Dict

from evalscope.api.registry import register_runner
from .base import AgentRunner, AgentRunResult, BridgeEndpoint, ExternalAgentTask, RunnerTimeoutError

# The body of the mock-agent CLI.  Lives as a string so it runs in any
# Python interpreter available inside the sandbox (no module import).
_MOCK_AGENT_SCRIPT = r"""
import json
import os
import sys
import urllib.request

def main() -> int:
    base_url = os.environ['ANTHROPIC_BASE_URL'].rstrip('/')
    token = os.environ['ANTHROPIC_AUTH_TOKEN']
    model = os.environ.get('ANTHROPIC_MODEL', 'mock-model')
    instruction = sys.stdin.read()

    body = {
        'model': model,
        'max_tokens': int(os.environ.get('MOCK_MAX_TOKENS', '1024')),
        'messages': [
            {'role': 'user', 'content': instruction},
        ],
    }
    req = urllib.request.Request(
        f'{base_url}/v1/messages',
        data=json.dumps(body).encode('utf-8'),
        headers={
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {token}',
            'anthropic-version': '2023-06-01',
        },
        method='POST',
    )
    with urllib.request.urlopen(req, timeout=float(os.environ.get('MOCK_TIMEOUT', '60'))) as resp:
        payload = json.loads(resp.read().decode('utf-8'))
    blocks = payload.get('content') or []
    text = ''.join(b.get('text', '') for b in blocks if isinstance(b, dict) and b.get('type') == 'text')
    sys.stdout.write(text)
    return 0

if __name__ == '__main__':
    sys.exit(main())
"""


@register_runner('mock')
class MockAgentRunner(AgentRunner):
    """One-shot Anthropic client used to validate the bridge end-to-end.

    Accepts ``model_name`` (forwarded to the bridge as ``ANTHROPIC_MODEL``)
    and ``max_tokens`` (forwarded as ``MOCK_MAX_TOKENS``) kwargs.
    """

    framework: str = 'mock'

    def __init__(
        self,
        *,
        model_name: str = '',
        max_tokens: int = 1024,
        **_: Any,
    ) -> None:
        self._model_name = model_name
        self._max_tokens = max_tokens

    async def setup(self, env: Any) -> None:
        # Nothing to install — script ships embedded.
        return None

    async def run(
        self,
        task: ExternalAgentTask,
        env: Any,
        bridge: BridgeEndpoint,
    ) -> AgentRunResult:
        env_vars: Dict[str, str] = {
            'ANTHROPIC_BASE_URL': f'{bridge.base_url}/anthropic',
            'ANTHROPIC_AUTH_TOKEN': bridge.trial_token,
            'MOCK_MAX_TOKENS': str(self._max_tokens),
        }
        if self._model_name:
            env_vars['ANTHROPIC_MODEL'] = self._model_name
        if task.timeout is not None:
            env_vars['MOCK_TIMEOUT'] = str(task.timeout)

        result = await env.exec(
            [sys.executable, '-c', _MOCK_AGENT_SCRIPT],
            input=task.instruction,
            timeout=task.timeout,
            env=env_vars,
        )
        if result.timed_out:
            raise RunnerTimeoutError(f'mock agent timed out after {task.timeout}s')
        if result.returncode != 0:
            stderr = (result.stderr or '').strip()
            raise RuntimeError(f'mock agent exited with code {result.returncode}: {stderr}')
        return AgentRunResult(
            output=result.stdout,
            metrics={
                'wall_time': result.duration,
                'returncode': result.returncode,
            },
        )
