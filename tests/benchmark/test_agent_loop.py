# Copyright (c) Alibaba, Inc. and its affiliates.
"""End-to-end evaluation tests for AgentLoop pipeline.

All tests run against the real qwen-plus API (DashScope).
No mocking – each test builds a TaskConfig and calls run_task().

Scenarios
---------
1. GSM8K baseline          – standard single-turn (no agent_config)
2. GSM8K agent, no tools   – function_calling strategy, pure multi-turn
3. GSM8K agent, python_exec + local env
4. GSM8K agent, bash + local env
5. GSM8K agent, python_exec + docker env
6. GSM8K + react strategy + python_exec + local env
7. GSM8K + swe_bench_toolcall strategy + bash + local env
"""

import json
import os
import shutil
import subprocess
import unittest
from dotenv import dotenv_values, load_dotenv
from pathlib import Path

load_dotenv('.env')
env = dotenv_values('.env')

from evalscope.api.agent import NativeAgentConfig
from evalscope.api.agent.trace import AgentTrace, EventType
from evalscope.config import TaskConfig
from evalscope.constants import EvalType
from evalscope.run import run_task
from evalscope.utils.logger import get_logger

logger = get_logger()

_API_KEY = env.get('DASHSCOPE_API_KEY')
_API_URL = 'https://dashscope.aliyuncs.com/compatible-mode/v1'

requires_api = unittest.skipUnless(_API_KEY, 'Requires DASHSCOPE_API_KEY in .env')


# ---------------------------------------------------------------------------
# Helpers for the external-agent (claude-code) e2e tests below
# ---------------------------------------------------------------------------

_IDEALAB_TOKEN = env.get('EVALSCOPE_IDEALAB_TOKEN') or os.environ.get('EVALSCOPE_IDEALAB_TOKEN')
_IDEALAB_URL = (
    env.get('EVALSCOPE_IDEALAB_BASE_URL') or os.environ.get('EVALSCOPE_IDEALAB_BASE_URL')
    or 'https://idealab.alibaba-inc.com/api/anthropic'
)
_IDEALAB_MODEL = env.get('EVALSCOPE_IDEALAB_MODEL') or os.environ.get('EVALSCOPE_IDEALAB_MODEL') or 'claude-opus-4-6'
_CLAUDE_CODE_IMAGE = (
    env.get('EVALSCOPE_CLAUDE_CODE_IMAGE') or os.environ.get('EVALSCOPE_CLAUDE_CODE_IMAGE')
    or 'evalscope/claude-code:dev'
)


def _docker_image_present(image: str) -> bool:
    """Cheap local probe for a docker image (no daemon call when CLI missing)."""
    if shutil.which('docker') is None:
        return False
    try:
        out = subprocess.run(
            ['docker', 'image', 'inspect', '--format', '{{.Id}}', image],
            capture_output=True, text=True, timeout=15,
        )
        return out.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


# Two-pronged gate: real LLM credentials AND the pre-baked claude-code image
# must be available. Build the image with
# ``bash evalscope/agent/external/runners/_assets/build_claude_code_image.sh``.
requires_claude_code_e2e = unittest.skipUnless(
    bool(_IDEALAB_TOKEN) and _docker_image_present(_CLAUDE_CODE_IMAGE),
    f'Requires EVALSCOPE_IDEALAB_TOKEN in .env and docker image {_CLAUDE_CODE_IMAGE!r} '
    f'(build via build_claude_code_image.sh).',
)

# The SWE-bench_Pro test needs a local clone of the upstream repo (auto-clone
# requires network). Reuses the cache the SWE-bench_Pro adapter would create.
_SWE_BENCH_PRO_REPO = Path.home() / '.cache' / 'evalscope' / 'swe_bench_pro' / 'SWE-bench_Pro-os'
requires_swe_bench_pro_repo = unittest.skipUnless(
    _SWE_BENCH_PRO_REPO.is_dir(),
    f'Requires SWE-bench_Pro-os clone at {_SWE_BENCH_PRO_REPO} '
    f'(see swe_bench_pro_repo_path docs).',
)


def _base_cfg(**overrides) -> dict:
    """Common TaskConfig kwargs shared by all tests."""
    cfg = {
        'model': 'qwen-plus',
        'api_url': _API_URL,
        'api_key': _API_KEY,
        'eval_type': EvalType.OPENAI_API,
        'eval_batch_size': 5,
        'limit': 5,
        'generation_config': {
            'max_tokens': 2048,
            'temperature': 0.7,
            'parallel_tool_calls': True,
        },
        'debug': True,
    }
    cfg.update(overrides)
    return cfg


def _read_review_results(benchmark: str, model: str = 'qwen-plus', work_dir: str = './outputs'):
    """Read all ReviewResult dicts from the JSONL review cache."""
    reviews_dir = Path(work_dir) / 'reviews' / model
    results = []
    for jsonl_file in reviews_dir.glob(f'{benchmark}_*.jsonl'):
        with open(jsonl_file) as f:
            for line in f:
                line = line.strip()
                if line:
                    results.append(json.loads(line))
    return results


# ---------------------------------------------------------------------------
# 1. GSM8K baseline (no agent_config)
# ---------------------------------------------------------------------------

@requires_api
class TestGSM8KBaseline(unittest.TestCase):
    """Standard single-turn GSM8K – no agent loop, verifies baseline works."""

    def test_gsm8k_no_agent(self):
        cfg = TaskConfig(
            **_base_cfg(
                datasets=['gsm8k'],
                dataset_args={'gsm8k': {'few_shot_num': 0}},
            )
        )
        result = run_task(cfg)
        self.assertIn('gsm8k', result)

    def test_gsm8k_no_agent_trace_absent(self):
        """ReviewResult.agent_trace must be None for non-agent runs."""
        cfg = TaskConfig(
            **_base_cfg(
                datasets=['gsm8k'],
                dataset_args={'gsm8k': {'few_shot_num': 0}},
            )
        )
        run_task(cfg)
        reviews = _read_review_results('gsm8k', work_dir=cfg.work_dir)
        self.assertGreater(len(reviews), 0)
        for r in reviews:
            self.assertIsNone(r.get('agent_trace'), 'agent_trace should be absent for baseline')


# ---------------------------------------------------------------------------
# 2. GSM8K + agent, no tools (pure multi-turn chain-of-thought)
# ---------------------------------------------------------------------------

@requires_api
class TestGSM8KAgentNoTools(unittest.TestCase):
    """function_calling strategy with no tools – model reasons step by step."""

    def test_run_succeeds_and_returns_score(self):
        cfg = TaskConfig(
            **_base_cfg(
                datasets=['gsm8k'],
                dataset_args={'gsm8k': {'few_shot_num': 0}},
                agent_config=NativeAgentConfig(
                    strategy='function_calling',
                    tools=[],
                    max_steps=3,
                ),
            )
        )
        result = run_task(cfg)
        self.assertIn('gsm8k', result)

# ---------------------------------------------------------------------------
# 3. GSM8K + agent, python_exec + local environment
# ---------------------------------------------------------------------------

@requires_api
class TestGSM8KAgentPythonExecLocal(unittest.TestCase):
    """function_calling strategy with python_exec tool + local subprocess env."""

    def test_run_and_trace_with_python_exec(self):
        """Run succeeds; trace.environment == 'local'; TOOL_CALL paired with TOOL_RESULT."""
        cfg = TaskConfig(
            **_base_cfg(
                datasets=['gsm8k'],
                dataset_args={'gsm8k': {'few_shot_num': 0}},
                agent_config=NativeAgentConfig(
                    strategy='function_calling',
                    tools=['python_exec'],
                    environment='local',
                    max_steps=5,
                    extra={'system_prompt': (
                        'Always call the python_exec tool to compute the answer.'
                    )},
                ),
            )
        )
        result = run_task(cfg)
        self.assertIn('gsm8k', result)

        reviews = _read_review_results('gsm8k', work_dir=cfg.work_dir)
        self.assertGreater(len(reviews), 0)
        any_tool_used = False
        for r in reviews:
            trace_dict = r.get('agent_trace')
            self.assertIsNotNone(trace_dict)
            trace = AgentTrace.model_validate(trace_dict)
            self.assertEqual(trace.environment, 'local')
            types = [e.type for e in trace.events]
            if EventType.TOOL_CALL in types:
                any_tool_used = True
                self.assertIn(EventType.TOOL_RESULT, types)
        logger.info(f'Any tool used: {any_tool_used}')
        # Note: we don't assert any_tool_used – model may choose not to call tool
        # but we verify the trace is structurally correct either way


# ---------------------------------------------------------------------------
# 4. GSM8K + agent, bash + local environment
# ---------------------------------------------------------------------------

@requires_api
class TestGSM8KAgentBashLocal(unittest.TestCase):
    """function_calling strategy with bash tool + local subprocess env."""

    def test_run_and_trace_with_bash(self):
        """Run succeeds; trace.environment == 'local' and events are present."""
        cfg = TaskConfig(
            **_base_cfg(
                datasets=['gsm8k'],
                dataset_args={'gsm8k': {'few_shot_num': 0}},
                agent_config=NativeAgentConfig(
                    strategy='function_calling',
                    tools=['bash'],
                    environment='local',
                    max_steps=5,
                ),
                eval_batch_size=1,
                limit=1,
            )
        )
        result = run_task(cfg)
        self.assertIn('gsm8k', result)

        reviews = _read_review_results('gsm8k', work_dir=cfg.work_dir)
        for r in reviews:
            trace = AgentTrace.model_validate(r['agent_trace'])
            self.assertEqual(trace.environment, 'local')
            self.assertGreater(len(trace.events), 0)


# ---------------------------------------------------------------------------
# 5. GSM8K + agent, python_exec + docker environment
# ---------------------------------------------------------------------------

@requires_api
class TestGSM8KAgentPythonExecDocker(unittest.TestCase):
    """function_calling strategy with python_exec tool + Docker sandbox env."""

    def test_run_and_trace_environment_is_docker(self):
        """Run succeeds, returns score, and AgentTrace.environment == 'docker'."""
        cfg = TaskConfig(
            **_base_cfg(
                datasets=['gsm8k'],
                dataset_args={'gsm8k': {'few_shot_num': 0}},
                agent_config=NativeAgentConfig(
                    strategy='function_calling',
                    tools=['python_exec'],
                    environment='docker',
                    environment_extra={'image': 'python:3.11-slim', 'timeout': 60},
                    max_steps=5,
                    extra={'system_prompt': (
                        'You are a math solver. '
                        'Use the python_exec tool to verify your calculations.'
                    )},
                ),
            )
        )
        result = run_task(cfg)
        self.assertIn('gsm8k', result)

        reviews = _read_review_results('gsm8k', work_dir=cfg.work_dir)
        self.assertGreater(len(reviews), 0)
        for r in reviews:
            trace_dict = r.get('agent_trace')
            self.assertIsNotNone(trace_dict)
            trace = AgentTrace.model_validate(trace_dict)
            self.assertEqual(trace.environment, 'docker')
            self.assertGreater(len(trace.events), 0)
            types = [e.type for e in trace.events]
            self.assertIn(EventType.MODEL_GENERATE, types)


# ---------------------------------------------------------------------------
# 6. GSM8K + react strategy + python_exec + local environment
# ---------------------------------------------------------------------------

@requires_api
class TestAIMEReActFC(unittest.TestCase):
    """ReAct strategy (FC mode) with bash tool + local env."""

    def test_react_run_and_trace(self):
        """Run succeeds; trace has strategy='react' and correct event structure."""
        cfg = TaskConfig(
            **_base_cfg(
                datasets=['aime26'],
                dataset_args={'aime26': {'few_shot_num': 0}},
                agent_config=NativeAgentConfig(
                    strategy='react',
                    tools=['bash'],
                    environment='local',
                    max_steps=10,
                ),
            )
        )
        result = run_task(cfg)
        self.assertIn('aime26', result)

        reviews = _read_review_results('aime26', work_dir=cfg.work_dir)
        self.assertGreater(len(reviews), 0)
        for r in reviews:
            trace_dict = r.get('agent_trace')
            self.assertIsNotNone(trace_dict)
            trace = AgentTrace.model_validate(trace_dict)
            self.assertEqual(trace.strategy, 'react')
            self.assertEqual(trace.environment, 'local')
            types = [e.type for e in trace.events]
            self.assertIn(EventType.MODEL_GENERATE, types)
            # ReAct FC mode should have TOOL_CALL events when tools are used.
            if EventType.TOOL_CALL in types:
                self.assertIn(EventType.TOOL_RESULT, types)


# ---------------------------------------------------------------------------
# 8. GSM8K + external agent (claude-code) inside the pre-baked docker image
# ---------------------------------------------------------------------------


class _ScrubAnthropicEnvMixin:
    """Strip ``ANTHROPIC_AUTH_TOKEN`` / ``ANTHROPIC_API_KEY`` /
    ``ANTHROPIC_BASE_URL`` for the duration of one test.

    Idealab rejects requests carrying both ``Authorization`` and
    ``x-api-key``; the Anthropic SDK auto-populates the former from
    ``ANTHROPIC_AUTH_TOKEN``, so any ``.env`` leakage breaks the test.
    """

    _SCRUB_KEYS = ('ANTHROPIC_AUTH_TOKEN', 'ANTHROPIC_API_KEY', 'ANTHROPIC_BASE_URL')

    def setUp(self) -> None:
        super().setUp()
        self._saved_env = {k: os.environ.pop(k, None) for k in self._SCRUB_KEYS}

    def tearDown(self) -> None:
        for k, v in self._saved_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        super().tearDown()


@requires_claude_code_e2e
class TestGSM8KExternalClaudeCode(_ScrubAnthropicEnvMixin, unittest.TestCase):
    """End-to-end: gsm8k → external claude-code CLI inside the pre-baked
    ``evalscope/claude-code:dev`` container → bridge → idealab Anthropic.

    Validates the full external-agent loop for a generic single-turn
    benchmark (no per-instance sandbox image needed; the runner image is
    the playground). Skipped unless the image is built AND idealab
    credentials are in ``.env``.
    """

    def test_gsm8k_through_external_claude_code(self):
        cfg = TaskConfig(
            model=_IDEALAB_MODEL,
            api_url=_IDEALAB_URL,
            api_key=_IDEALAB_TOKEN,
            eval_type='anthropic_api',
            # Idealab User-Agent whitelist: only claude-cli traffic passes.
            model_args={
                'default_headers': {
                    'User-Agent': 'claude-cli/2.1.146 (external, cli)',
                },
            },
            datasets=['gsm8k'],
            dataset_args={'gsm8k': {'few_shot_num': 0}},
            agent_config={
                'mode': 'external',
                'framework': 'claude-code',
                'environment': 'docker',
                # The pre-baked image already has node + claude on PATH;
                # ``auto_install=False`` short-circuits ``setup()``'s probe.
                'environment_extra': {
                    'sandbox_config': {
                        'image': _CLAUDE_CODE_IMAGE,
                        'working_dir': '/workspace',
                    },
                    'timeout': 180.0,
                },
                'kwargs': {
                    'auto_install': False,
                    # Empty string = inherit the inside-container HOME.
                    'home_override': '',
                },
                'timeout': 300.0,
            },
            eval_batch_size=1,
            limit=1,
        )
        result = run_task(cfg)
        self.assertIn('gsm8k', result)

        # Trace must reflect the external-agent path (framework='claude-code',
        # environment='docker') — proves the external dispatch fired and the
        # docker env flowed through.
        reviews = _read_review_results('gsm8k', model=_IDEALAB_MODEL, work_dir=cfg.work_dir)
        self.assertGreater(len(reviews), 0)
        for r in reviews:
            trace_dict = r.get('agent_trace')
            self.assertIsNotNone(trace_dict, 'external-agent path must populate agent_trace')
            trace = AgentTrace.model_validate(trace_dict)
            self.assertEqual(trace.framework, 'claude-code')
            # EnclaveAgentEnvironment's instance ``name`` is always 'enclave'
            # regardless of the 'docker' / 'volcengine' registry alias used.
            self.assertEqual(trace.environment, 'enclave')
            types = [e.type for e in trace.events]
            self.assertIn(EventType.RUN_START, types)
            self.assertIn(EventType.MODEL_GENERATE, types)
            self.assertIn(EventType.RUN_END, types)


# ---------------------------------------------------------------------------
# 9. SWE-bench_Pro + external agent (claude-code) — full pipeline including
#    eval container + scored Report. Pins that evaluation logs / artifacts
#    look normal (predictions / reviews / reports directories populated,
#    swe_bench_pro_log artefacts written, score numeric).
# ---------------------------------------------------------------------------


@requires_claude_code_e2e
@requires_swe_bench_pro_repo
class TestSWEBenchProExternalClaudeCode(_ScrubAnthropicEnvMixin, unittest.TestCase):
    """End-to-end SWE-bench_Pro through external claude-code.

    Slow / expensive: pulls a per-instance sweap-image (~5–15GB),
    installs claude-code at runtime, runs claude against a real LLM,
    then runs the SWE-bench_Pro evaluation container.

    Asserts the pipeline *executes* and writes its normal evaluation
    artifacts (predictions / reviews / reports / swe_bench_pro_log) —
    not the resolved flag, which is model-quality dependent.
    """

    def test_swe_bench_pro_through_external_claude_code(self):
        cfg = TaskConfig(
            model=_IDEALAB_MODEL,
            api_url=_IDEALAB_URL,
            api_key=_IDEALAB_TOKEN,
            eval_type='anthropic_api',
            model_args={
                'default_headers': {
                    'User-Agent': 'claude-cli/2.1.146 (external, cli)',
                },
            },
            datasets=['swe_bench_pro'],
            dataset_args={
                'swe_bench_pro': {
                    'extra_params': {
                        # Skip the auto-clone + ``git fetch`` (flaky on
                        # locked-down corp networks); the @requires_swe_bench_pro_repo
                        # guard already asserts the path exists.
                        'swe_bench_pro_repo_path': str(_SWE_BENCH_PRO_REPO),
                    },
                },
            },
            agent_config={
                'mode': 'external',
                'framework': 'claude-code',
                # ``environment`` / ``environment_extra`` are intentionally
                # absent: AgentLoopAdapter uses the benchmark's own
                # ``build_environment(sample)`` (per-instance sweap-image).
                # sweap-images ship Node out of the box; auto_install
                # auto-detects and skips apt+nodesource (only
                # ``npm install -g claude-code`` runs, ~20s).
                'kwargs': {},
                'timeout': 1500.0,
            },
            # NodeBB (and other heavy JS samples) spawn Redis + jest under
            # the agent's bash steps and OOM with the docker default. The
            # SWE-bench_Pro adapter explicitly documents this as the fix.
            sandbox={
                'default_config': {
                    'memory_limit': '8g',
                    'cpu_limit': 4,
                },
            },
            # Don't let one OOM'd sample short-circuit the whole batch —
            # this test needs both samples to write predictions/reviews
            # to validate the multi-sample path.
            ignore_errors=True,
            eval_batch_size=2,
            limit=2,
        )
        result = run_task(cfg)
        self.assertIn('swe_bench_pro', result)

        # Evaluation log artifacts must be present — pins that the
        # adapter's ``match_score`` actually ran ``eval_instance`` inside
        # the secondary container (the part that signals "evaluation
        # logging is normal").
        work_dir = Path(cfg.work_dir)
        swe_logs = work_dir / 'swe_bench_pro_log'
        self.assertTrue(
            swe_logs.is_dir(),
            f'expected swe_bench_pro_log under {work_dir} — eval_instance did not run',
        )
        # At least one instance directory with an output.json (or stderr.log).
        instance_dirs = [p for p in swe_logs.iterdir() if p.is_dir()]
        self.assertGreater(len(instance_dirs), 0, 'no per-instance log directory written')
        for inst_dir in instance_dirs:
            workspace = inst_dir / 'workspace'
            self.assertTrue(workspace.is_dir(), f'no workspace under {inst_dir}')
            # The eval entry script + parser are always staged; the
            # patch may be empty if the agent did nothing.
            self.assertTrue((workspace / 'entryscript.sh').is_file())
            self.assertTrue((workspace / 'patch.diff').is_file())

        # Standard EvalScope outputs.
        for sub in ('predictions', 'reviews', 'reports'):
            self.assertTrue(
                (work_dir / sub).is_dir(),
                f'standard output dir {sub!r} missing under {work_dir}',
            )

        # Trace surfaces in the review with the right framework / env.
        reviews = _read_review_results('swe_bench_pro', model=_IDEALAB_MODEL, work_dir=cfg.work_dir)
        self.assertGreater(len(reviews), 0)
        for r in reviews:
            trace_dict = r.get('agent_trace')
            self.assertIsNotNone(trace_dict)
            trace = AgentTrace.model_validate(trace_dict)
            self.assertEqual(trace.framework, 'claude-code')
            self.assertEqual(trace.environment, 'enclave')


# ---------------------------------------------------------------------------
# 10. SWE-bench_Pro + external agent (codex) → bridge Responses API
#     → DashScope qwen3-max. Mirrors the claude-code variant above but
#     drives the OpenAI Responses route (PR2) end-to-end.
# ---------------------------------------------------------------------------


@requires_api
@requires_swe_bench_pro_repo
class TestSWEBenchProExternalCodex(unittest.TestCase):
    """End-to-end SWE-bench_Pro through external codex CLI × qwen3-max.

    Differences vs ``TestSWEBenchProExternalClaudeCode``:

    * ``framework='codex'`` → bridge ``POST /openai/v1/responses`` route
      (codex v0.133+ refuses chat completions).
    * Backend is DashScope qwen3-max via the OpenAI-compatible endpoint;
      no idealab User-Agent whitelist needed.
    * No ``_ScrubAnthropicEnvMixin`` — codex talks OpenAI; no Anthropic
      SDK env-var conflict can arise.
    * No pre-baked image gate: ``CodexRunner.setup`` installs
      ``@openai/codex`` via apt+nodesource+npm inside the per-instance
      sweap-image (~30s amortised on first sample of a series; previously
      verified end-to-end in PR2's NodeBB 37-turn run).

    Slow / expensive: pulls per-instance sweap-images (~5-15 GB each),
    runs codex × qwen3-max against the real DashScope API, then runs the
    SWE-bench_Pro evaluation container. Budget per sample: 5-20 min wall,
    ¥0.5-5 in DashScope tokens.
    """

    def test_swe_bench_pro_through_external_codex(self):
        cfg = TaskConfig(
            model='qwen3-max',
            api_url=_API_URL,
            api_key=_API_KEY,
            eval_type=EvalType.OPENAI_API,
            datasets=['swe_bench_pro'],
            dataset_args={
                'swe_bench_pro': {
                    'extra_params': {
                        # Same auto-clone-skip as the claude-code variant.
                        'swe_bench_pro_repo_path': str(_SWE_BENCH_PRO_REPO),
                    },
                },
            },
            agent_config={
                'mode': 'external',
                'framework': 'codex',
                # AgentLoopAdapter uses the benchmark's per-instance
                # sweap-image; no environment / environment_extra override
                # (parity with the claude-code variant). Kwargs are empty:
                # CodexRunner defaults already cover the SWE-bench Pro path
                # (sandbox=workspace-write hardcoded, non-interactive,
                # auto-install enabled, model_name auto-inherited from
                # TaskConfig.model).
                'kwargs': {},
                'timeout': 1500.0,
            },
            # Same memory / cpu fix as the claude-code variant — codex's
            # apply_patch + exec_command can spawn the same heavy JS test
            # stacks (Redis, jest, etc.) under workspace-write.
            sandbox={
                'default_config': {
                    'memory_limit': '8g',
                    'cpu_limit': 4,
                },
            },
            ignore_errors=True,
            eval_batch_size=3,
            limit=3,
        )
        result = run_task(cfg)
        self.assertIn('swe_bench_pro', result)

        # Pin the same evaluation-log structure the claude-code variant pins.
        work_dir = Path(cfg.work_dir)
        swe_logs = work_dir / 'swe_bench_pro_log'
        self.assertTrue(
            swe_logs.is_dir(),
            f'expected swe_bench_pro_log under {work_dir} — eval_instance did not run',
        )
        instance_dirs = [p for p in swe_logs.iterdir() if p.is_dir()]
        self.assertGreater(len(instance_dirs), 0, 'no per-instance log directory written')
        for inst_dir in instance_dirs:
            workspace = inst_dir / 'workspace'
            self.assertTrue(workspace.is_dir(), f'no workspace under {inst_dir}')
            self.assertTrue((workspace / 'entryscript.sh').is_file())
            self.assertTrue((workspace / 'patch.diff').is_file())

        for sub in ('predictions', 'reviews', 'reports'):
            self.assertTrue(
                (work_dir / sub).is_dir(),
                f'standard output dir {sub!r} missing under {work_dir}',
            )

        # Trace must reflect the external-agent codex path.
        reviews = _read_review_results('swe_bench_pro', model='qwen3-max', work_dir=cfg.work_dir)
        self.assertGreater(len(reviews), 0)
        for r in reviews:
            trace_dict = r.get('agent_trace')
            self.assertIsNotNone(trace_dict, 'external-agent path must populate agent_trace')
            trace = AgentTrace.model_validate(trace_dict)
            self.assertEqual(trace.framework, 'codex')
            self.assertEqual(trace.environment, 'enclave')
            types = [e.type for e in trace.events]
            self.assertIn(EventType.RUN_START, types)
            self.assertIn(EventType.MODEL_GENERATE, types)
            self.assertIn(EventType.RUN_END, types)


if __name__ == '__main__':
    unittest.main()
