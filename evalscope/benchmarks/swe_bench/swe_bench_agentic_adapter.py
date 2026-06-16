"""SWE-bench agentic adapter (mini-swe-agent compatible).

Drives a multi-turn :class:`AgentLoop` per sample with a per-instance
SWE-bench Docker container as the execution sandbox.  Mirrors the original
``mini-swe-agent`` ``swebench.yaml`` (toolcall mainline) and
``swebench_backticks.yaml`` (textbased fallback) configurations.

Three benchmarks are registered alongside the original oracle adapters
(without disrupting them):

- ``swe_bench_verified_agentic``
- ``swe_bench_verified_mini_agentic``
- ``swe_bench_lite_agentic``

The original ``swe_bench_verified`` / ``swe_bench_verified_mini`` /
``swe_bench_lite`` benchmarks remain single-turn oracle-text evaluations
and are not affected.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from evalscope.agent.tools.bash import BASH_TOOL_INFO, run_bash
from evalscope.api.agent import AgentEnvironment, AgentStrategy
from evalscope.api.benchmark import BenchmarkMeta
from evalscope.api.benchmark.adapters import AgentLoopAdapter
from evalscope.api.dataset import FieldSpec, RemoteDataLoader, Sample
from evalscope.api.evaluator import TaskState
from evalscope.api.messages import ChatMessageUser
from evalscope.api.metric import Score
from evalscope.api.registry import register_benchmark
from evalscope.constants import Tags
from evalscope.utils.import_utils import check_import, is_build_doc
from evalscope.utils.logger import get_logger

if TYPE_CHECKING:
    from evalscope.agent.external.runners import AgentRunResult

logger = get_logger()

# ---------------------------------------------------------------------------
# instance_template — mirrors mini-swe-agent swebench.yaml
# ---------------------------------------------------------------------------

INSTANCE_TEMPLATE = """\
<pr_description>
Consider the following PR description:
{problem_statement}
</pr_description>

<instructions>
# Task Instructions

## Overview

You're a software engineer interacting continuously with a computer by submitting commands.
You'll be helping implement necessary changes to meet requirements in the PR description.
Your task is specifically to make changes to non-test files in the current directory in order to fix the issue described in the PR description in a way that is general and consistent with the codebase.
<IMPORTANT>This is an interactive process where you will think and issue AT LEAST ONE command, see the result, then think and issue your next command(s).</important>

For each response:

1. Include a THOUGHT section explaining your reasoning and what you're trying to accomplish
2. Provide one or more bash tool calls to execute

## Important Boundaries

- MODIFY: Regular source code files in /testbed (this is the working directory for all your subsequent commands)
- DO NOT MODIFY: Tests, configuration files (pyproject.toml, setup.cfg, etc.)

## Recommended Workflow

1. Analyze the codebase by finding and reading relevant files
2. Create a script to reproduce the issue
3. Edit the source code to resolve the issue
4. Verify your fix works by running your script again
5. Test edge cases to ensure your fix is robust

## Command Execution Rules

You are operating in an environment where

1. You issue at least one command
2. The system executes the command(s) in a subshell
3. You see the result(s)
4. You write your next command(s)

Each response should include:

1. **Reasoning text** where you explain your analysis and plan
2. At least one tool call with your command

**CRITICAL REQUIREMENTS:**

- Your response SHOULD include reasoning text explaining what you're doing
- Your response MUST include AT LEAST ONE bash tool call. You can make MULTIPLE tool calls in a single response when the commands are independent (e.g., searching multiple files, reading different parts of the codebase).
- Directory or environment variable changes are not persistent. Every action is executed in a new subshell.
- However, you can prefix any action with `MY_ENV_VAR=MY_VALUE cd /path/to/working/dir && ...` or write/load environment variables from files

Example of a CORRECT response:
<example_response>
I need to understand the Builder-related code. Let me find relevant files and check the project structure.

[Makes multiple bash tool calls: {{"command": "ls -la"}}, {{"command": "find src -name '*.java' | grep -i builder"}}, {{"command": "cat README.md | head -50"}}]
</example_response>

## Environment Details

- You have a full Linux shell environment
- Always use non-interactive flags (-y, -f) for commands
- Avoid interactive tools like vi, nano, or any that require user input
- You can use bash commands or invoke any tool that is available in the environment
- You can also create new tools or scripts to help you with the task
- If a tool isn't available, you can also install it

## Submission

When you've completed your work, you MUST submit your changes as a git patch.
Follow these steps IN ORDER, with SEPARATE commands:

Step 1: Create the patch file
Run `git diff -- path/to/file1 path/to/file2 > patch.txt` listing only the source files you modified.
Do NOT commit your changes.

<IMPORTANT>
The patch must only contain changes to the specific source files you modified to fix the issue.
Do not submit file creations or changes to any of the following files:

- test and reproduction files
- helper scripts, tests, or tools that you created
- installation, build, packaging, configuration, or setup scripts unless they are directly part of the issue you were fixing (you can assume that the environment is already set up for your client)
- binary or compiled files
</IMPORTANT>

Step 2: Verify your patch
Inspect patch.txt to confirm it only contains your intended changes and headers show `--- a/` and `+++ b/` paths.

Step 3: Submit (EXACT command required)
You MUST use this EXACT command to submit:

```bash
echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && cat patch.txt
```

If the command fails (nonzero exit status), it will not submit.

<CRITICAL>
- Creating/viewing the patch and submitting it MUST be separate commands (not combined with &&).
- If you modify patch.txt after verifying, you SHOULD verify again before submitting.
- You CANNOT continue working (reading, editing, testing) in any way on this task after submitting.
</CRITICAL>
</instructions>
"""

# ---------------------------------------------------------------------------
# Shared extra_params definitions
# ---------------------------------------------------------------------------

_AGENTIC_EXTRA_PARAMS: Dict[str, Any] = {
    'action_protocol': {
        'type': 'str',
        'description': (
            'Agent action protocol: "toolcall" (mainline OpenAI '
            'function-calling, mirrors mini-swe-agent swebench.yaml) or '
            '"backticks" (textbased mswea_bash_command fallback for models '
            'without function-calling support).'
        ),
        'value': 'toolcall',
        'choices': ['toolcall', 'backticks'],
    },
    'max_steps': {
        'type': 'int',
        'description': 'Maximum number of agent steps per sample.',
        'value': 250,
    },
    'command_timeout': {
        'type': 'float',
        'description': 'Default per-bash-command timeout in seconds.',
        'value': 60.0,
    },
    'build_docker_images': {
        'type': 'bool',
        'description': 'Build Docker images locally for each sample.',
        'value': True,
    },
    'pull_remote_images_if_available': {
        'type': 'bool',
        'description': 'Attempt to pull existing remote Docker images before building.',
        'value': True,
    },
    'force_arch': {
        'type': 'str',
        'description': 'Optionally force a specific architecture for image build/pull.',
        'value': '',
        'choices': ['', 'arm64', 'x86_64'],
    },
    'dockerhub_username': {
        'type': 'str',
        'description': 'DockerHub user/org namespace for remote SWE-bench images.',
        'value': 'swebench',
    },
}

# ---------------------------------------------------------------------------
# Base adapter
# ---------------------------------------------------------------------------


class _SWEBenchAgenticAdapterBase(AgentLoopAdapter):
    """Shared agentic SWE-bench adapter logic.

    Concrete benchmarks differ only in dataset_id / pretty_name; behaviour
    is identical otherwise.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

        check_import('swebench', extra='swe_bench', raise_error=True, feature_name=self.pretty_name)

        self.action_protocol: str = self.extra_params.get('action_protocol', 'toolcall')
        if self.action_protocol not in {'toolcall', 'backticks'}:
            raise ValueError(
                f'Invalid action_protocol={self.action_protocol!r}; '
                "must be 'toolcall' or 'backticks'."
            )
        self.max_steps = int(self.extra_params.get('max_steps', 250))
        self.command_timeout = float(self.extra_params.get('command_timeout', 60.0))
        # Hardcoded: must match the /testbed path used by swebench harness eval_script.
        self.working_dir: str = '/testbed'
        self.build_docker_images: bool = self.extra_params.get('build_docker_images', True)
        self.pull_remote_images_if_available: bool = self.extra_params.get('pull_remote_images_if_available', True)
        self.force_arch: str = self.extra_params.get('force_arch', '')
        self.dockerhub_username: str = self.extra_params.get('dockerhub_username') or 'swebench'

        # Skip docker image build/pull during documentation generation (BUILD_DOC=1)
        # to avoid slow/unnecessary image pulls when running `make docs-pipeline`.
        if is_build_doc():
            self.build_docker_images = False
            self.pull_remote_images_if_available = False

    # ------------------------------------------------------------------
    # Dataset loading
    # ------------------------------------------------------------------

    def record_to_sample(self, record: Dict[str, Any]) -> Sample:
        # Agentic mode never uses oracle ``text``; the model must explore
        # /testbed itself.  Initial user message is constructed in
        # ``build_initial_messages`` from ``problem_statement``.
        return Sample(
            input=record['problem_statement'],
            metadata={
                'problem_statement': record['problem_statement'],
                'instance_id': record['instance_id'],
                'base_commit': record['base_commit'],
                'patch': record['patch'],
                'PASS_TO_PASS': json.loads(record['PASS_TO_PASS']),
                'FAIL_TO_PASS': json.loads(record['FAIL_TO_PASS']),
                'test_patch': record['test_patch'],
                'version': record['version'],
                'repo': record['repo'],
                'environment_setup_commit': record['environment_setup_commit'],
                'hints_text': record['hints_text'],
                'created_at': record['created_at'],
            },
        )

    def _post_process_samples(self):
        """Ensure each sample carries its per-instance Docker image."""
        from .build_images import build_images
        from .utils import get_remote_docker_image_from_id

        samples = self.test_dataset[self.default_subset]

        if self.build_docker_images:
            id_to_docker_image_map = build_images(
                samples=samples,
                force_rebuild=False,
                max_workers=4,
                use_remote_images=self.pull_remote_images_if_available,
                force_arch=self.force_arch,
                dockerhub_username=self.dockerhub_username,
            )

            def docker_image_from_id(instance_id: str) -> str:
                return id_to_docker_image_map.get(instance_id, '')
        else:
            from functools import partial
            docker_image_from_id = partial(
                get_remote_docker_image_from_id,
                dockerhub_username=self.dockerhub_username,
                force_arch=self.force_arch,
            )

        for sample in samples:
            sample.metadata['docker_image'] = docker_image_from_id(sample.metadata['instance_id'])
            existing_tools = list(sample.tools or [])
            if not any(t.name == 'bash' for t in existing_tools):
                existing_tools.append(BASH_TOOL_INFO)
            sample.tools = existing_tools

        super()._post_process_samples()

    # ------------------------------------------------------------------
    # AgentAdapter hooks
    # ------------------------------------------------------------------

    def build_strategy(self, sample: Sample) -> AgentStrategy:
        if self.action_protocol == 'toolcall':
            from evalscope.agent.strategies.swe_bench import SweBenchToolcallStrategy
            return SweBenchToolcallStrategy()
        from evalscope.agent.strategies.swe_bench import SweBenchBackticksStrategy
        return SweBenchBackticksStrategy()

    def build_tools(self, sample: Sample):
        # Only ``bash`` — sentinel protocol replaces the ``submit`` tool.
        return {'bash': run_bash}

    def build_environment(self, sample: Sample) -> Optional[AgentEnvironment]:
        from evalscope.agent.environments.enclave import EnclaveAgentEnvironment

        image = sample.metadata.get('docker_image')
        if not image:
            raise RuntimeError(
                f"docker_image missing for instance {sample.metadata.get('instance_id')!r}; "
                'did _post_process_samples run?'
            )

        sandbox_config = {
            'image': image,
            'working_dir': self.working_dir,
            'environment': {
                'PAGER': 'cat',
                'MANPAGER': 'cat',
                'LESS': '-R',
                'PIP_PROGRESS_BAR': 'off',
                'TQDM_DISABLE': '1',
            },
        }
        return EnclaveAgentEnvironment(
            engine='docker',
            sandbox_config=sandbox_config,
            timeout=self.command_timeout,
        )

    def build_initial_messages(self, sample: Sample) -> List[Any]:
        problem_statement = sample.metadata.get('problem_statement'
                                                ) or (sample.input if isinstance(sample.input, str) else '')
        rendered = INSTANCE_TEMPLATE.format(problem_statement=problem_statement)
        return [ChatMessageUser(content=rendered)]

    # ------------------------------------------------------------------
    # External agent prediction (recover patch via git diff)
    # ------------------------------------------------------------------

    async def _external_extract_prediction(
        self, env: AgentEnvironment, run_result: AgentRunResult, sample: Sample
    ) -> str:
        """Recover the agent's patch from ``self.working_dir`` (``/testbed``).

        External CLI agents do not implement the
        ``COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT`` sentinel protocol, so
        the patch is reconstructed from the working tree against
        ``HEAD`` instead. The returned diff is fed straight into
        :meth:`match_score`'s ``eval_instance``, reusing the entire
        SWE-bench evaluation pipeline.
        """
        from evalscope.agent.external.helpers import extract_patch
        return await extract_patch(env, cwd=self.working_dir)

    # ------------------------------------------------------------------
    # Final answer extraction
    # ------------------------------------------------------------------

    def _extract_final_answer(self, result, strategy: AgentStrategy) -> str:
        # Strategy implements sentinel scanning by mutating
        # ``ParsedAction.final_answer`` from ``format_observation`` and
        # archiving the raw submission payload as the last tool/user
        # message; ``strategy.extract_final_answer`` recovers it from
        # ``result.messages``.
        answer = strategy.extract_final_answer(result)
        if answer:
            return answer
        # Fallback: try to recover a unified-diff from the last assistant
        # message (model didn't follow the sentinel protocol).
        try:
            from swebench.inference.make_datasets.utils import extract_diff

            last_assistant = ''
            for msg in reversed(result.messages):
                if msg.role == 'assistant':
                    last_assistant = str(msg.content or '') or msg.text or ''
                    if last_assistant:
                        break
            return extract_diff(last_assistant) or ''
        except Exception:  # pragma: no cover - defensive
            return ''

    def extract_answer(self, prediction: str, task_state) -> str:
        """Return the prediction as-is; sentinel extraction is authoritative."""
        if prediction:
            return prediction
        from swebench.inference.make_datasets.utils import extract_diff
        return extract_diff(prediction or '')

    # ------------------------------------------------------------------
    # Scoring (re-uses the existing oracle eval pipeline)
    # ------------------------------------------------------------------

    def match_score(
        self,
        original_prediction: str,
        filtered_prediction: str,
        reference: str,
        task_state: TaskState,
    ) -> Score:
        from .utils import eval_instance

        score = Score(
            extracted_prediction=filtered_prediction,
            prediction=original_prediction,
        )

        result = eval_instance(
            instance=task_state.metadata,
            pred=filtered_prediction,
            timeout=1800,
            log_dir=self._task_config.work_dir,
            dockerhub_username=self.dockerhub_username,
            force_arch=self.force_arch,
        )

        score.value = {'acc': float(result.get('resolved', 0.0))}
        score.metadata = result
        return score


# ---------------------------------------------------------------------------
# Concrete benchmarks
# ---------------------------------------------------------------------------

_AGENTIC_MODE_SECTION = """
## Agentic Mode

This benchmark drives a multi-turn agent loop (mirrors mini-swe-agent's
`swebench.yaml`) inside a per-instance SWE-bench Docker container. The
model issues `bash` commands to explore `/testbed`, edit source files,
and finally submits its `git diff` patch by printing the sentinel
`COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT` followed by the patch contents.

`extra_params.action_protocol` selects between:
- `toolcall` (default): OpenAI function-calling protocol with a single
  `bash` tool. Recommended for any model that supports tool calling.
- `backticks`: text-based fallback expecting one
  ` ```mswea_bash_command ``` ` block per turn. For models without
  function-calling support.
"""

_SWE_BENCH_VERIFIED_AGENTIC_DESCRIPTION = """
## Overview

SWE-bench Verified Agentic is the agentic-mode evaluation of SWE-bench Verified, a human-validated subset of 500 samples from SWE-bench. Unlike the oracle single-turn variant, the model must autonomously explore the repository, run shell commands, edit source files, and submit a patch through a multi-turn agent loop driven inside a per-instance Docker container.

## Task Description

- **Task Type**: Automated Software Engineering / Bug Fixing (Agentic)
- **Input**: GitHub issue description (no oracle file context)
- **Output**: Code patch (diff format) collected from `git diff` after autonomous editing
- **Repositories**: 12 popular Python projects (Django, Flask, Requests, etc.)

## Key Features

- 500 human-validated Issue-Pull Request pairs
- Multi-turn agent loop (mini-swe-agent `swebench.yaml` compatible)
- Per-instance SWE-bench Docker container as the execution sandbox
- Sentinel-based patch submission protocol (`COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT`)
- Supports both function-calling (`toolcall`) and text-based (`backticks`) action protocols

## Evaluation Notes

- Requires `pip install swebench==4.1.0` before evaluation
- Docker images are built/pulled automatically for each repository
- Timeout of 1800 seconds (30 min) per instance for final patch validation
- See the [usage documentation](https://evalscope.readthedocs.io/en/latest/third_party/swe_bench.html) for detailed setup instructions
- Supports both local image building and remote image pulling
""" + _AGENTIC_MODE_SECTION

_SWE_BENCH_VERIFIED_MINI_AGENTIC_DESCRIPTION = """
## Overview

SWE-bench Verified Mini Agentic is the agentic-mode evaluation of SWE-bench Verified Mini, a compact 50-sample subset that maintains the same distribution of performance, test pass rates, and difficulty as the full Verified set while requiring only 5GB of storage instead of 130GB. The model must autonomously explore, edit, and submit a patch through a multi-turn agent loop.

## Task Description

- **Task Type**: Automated Software Engineering / Bug Fixing (Agentic)
- **Input**: GitHub issue description (no oracle file context)
- **Output**: Code patch (diff format) collected from `git diff` after autonomous editing
- **Size**: 50 samples (vs 500 in full Verified set)

## Key Features

- Representative 50-sample subset of SWE-bench Verified
- Same difficulty distribution as the full dataset
- Dramatically reduced storage requirements (5GB vs 130GB)
- Multi-turn agent loop with per-instance Docker sandbox
- Ideal for quick agentic evaluation and development iteration

## Evaluation Notes

- Requires `pip install swebench==4.1.0` before evaluation
- Docker images are built/pulled automatically
- See the [usage documentation](https://evalscope.readthedocs.io/en/latest/third_party/swe_bench.html) for detailed setup
- Good for rapid prototyping of agent strategies and initial model assessment
""" + _AGENTIC_MODE_SECTION

_SWE_BENCH_LITE_AGENTIC_DESCRIPTION = """
## Overview

SWE-bench Lite Agentic is the agentic-mode evaluation of SWE-bench Lite, a focused subset of SWE-bench containing 300 Issue-Pull Request pairs from 11 popular Python repositories. The model autonomously drives a multi-turn agent loop inside a per-instance Docker container to resolve real-world GitHub issues.

## Task Description

- **Task Type**: Automated Software Engineering / Bug Fixing (Agentic)
- **Input**: GitHub issue description (no oracle file context)
- **Output**: Code patch (diff format) collected from `git diff` after autonomous editing
- **Size**: 300 carefully selected test instances

## Key Features

- 300 test Issue-Pull Request pairs
- 11 popular Python repositories covered
- Real-world bugs with verified solutions
- Multi-turn agent loop with per-instance Docker sandbox
- More manageable than full SWE-bench while still challenging

## Evaluation Notes

- Requires `pip install swebench==4.1.0` before evaluation
- Docker images are built/pulled automatically for each repository
- See the [usage documentation](https://evalscope.readthedocs.io/en/latest/third_party/swe_bench.html) for detailed setup instructions
- Popular benchmark variant for initial agentic model comparison
""" + _AGENTIC_MODE_SECTION


@register_benchmark(
    BenchmarkMeta(
        name='swe_bench_verified_agentic',
        pretty_name='SWE-bench_Verified_Agentic',
        tags=[Tags.CODING],
        description=_SWE_BENCH_VERIFIED_AGENTIC_DESCRIPTION,
        dataset_id='princeton-nlp/SWE-bench_Verified',
        metric_list=['acc'],
        eval_split='test',
        prompt_template='{question}',
        extra_params=_AGENTIC_EXTRA_PARAMS,
    )
)
class SWEBenchVerifiedAgenticAdapter(_SWEBenchAgenticAdapterBase):
    ...


@register_benchmark(
    BenchmarkMeta(
        name='swe_bench_verified_mini_agentic',
        pretty_name='SWE-bench_Verified_Mini_Agentic',
        tags=[Tags.CODING],
        description=_SWE_BENCH_VERIFIED_MINI_AGENTIC_DESCRIPTION,
        dataset_id='evalscope/swe-bench-verified-mini',
        metric_list=['acc'],
        eval_split='test',
        prompt_template='{question}',
        extra_params=_AGENTIC_EXTRA_PARAMS,
    )
)
class SWEBenchVerifiedMiniAgenticAdapter(_SWEBenchAgenticAdapterBase):
    ...


@register_benchmark(
    BenchmarkMeta(
        name='swe_bench_lite_agentic',
        pretty_name='SWE-bench_Lite_Agentic',
        tags=[Tags.CODING],
        description=_SWE_BENCH_LITE_AGENTIC_DESCRIPTION,
        dataset_id='princeton-nlp/SWE-bench_Lite',
        metric_list=['acc'],
        eval_split='test',
        prompt_template='{question}',
        extra_params=_AGENTIC_EXTRA_PARAMS,
    )
)
class SWEBenchLiteAgenticAdapter(_SWEBenchAgenticAdapterBase):
    ...
