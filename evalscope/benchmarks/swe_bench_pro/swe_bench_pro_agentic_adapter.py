"""SWE-bench_Pro agentic adapter.

Drives a multi-turn AgentLoop per sample with a per-instance Docker
container (``jefzda/sweap-images:{tag}``) as the execution sandbox.
The model issues ``bash`` commands inside ``/app``, edits files, and
submits its ``git diff`` patch via the ``COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT``
sentinel (mirrors the ``swe_bench_*_agentic`` benchmarks for Verified/Lite).

Eval flow inside the container (see :func:`utils.eval_instance`):
``git reset --hard {base_commit} && git apply patch.diff &&
{before_repo_set_cmd} && bash run_script.sh {test_files} && python parser.py``
then check ``(fail_to_pass | pass_to_pass) ⊆ PASSED``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional

from evalscope.agent.tools.bash import BASH_TOOL_INFO, run_bash
from evalscope.api.agent import AgentEnvironment, AgentStrategy
from evalscope.api.benchmark import BenchmarkMeta
from evalscope.api.benchmark.adapters import AgentLoopAdapter
from evalscope.api.dataset import Sample
from evalscope.api.evaluator import TaskState
from evalscope.api.messages import ChatMessageUser
from evalscope.api.metric import Score
from evalscope.api.registry import register_benchmark
from evalscope.api.sandbox import merge_sandbox_config_dicts
from evalscope.constants import Tags
from evalscope.utils.import_utils import is_build_doc
from evalscope.utils.logger import get_logger

if TYPE_CHECKING:
    from evalscope.agent.external.runners import AgentRunResult

logger = get_logger()

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

- MODIFY: Regular source code files in /app (this is the working directory for all your subsequent commands)
- DO NOT MODIFY: Tests, configuration files (package.json, pyproject.toml, setup.cfg, etc.)

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

_DESCRIPTION = """
## Overview

SWE-bench_Pro is a challenging benchmark from Scale AI evaluating LLMs/Agents on long-horizon software engineering tasks across multiple programming languages. Given a codebase and an issue, the model must autonomously explore the repository, edit source files, and submit a patch through a multi-turn agent loop inside a per-instance Docker container.

## Task Description

- **Task Type**: Automated Software Engineering / Bug Fixing (Agentic)
- **Input**: GitHub issue description
- **Output**: Code patch (diff format) collected after autonomous editing
- **Languages**: Multiple (`repo_language` field; e.g. JavaScript/TypeScript, Python, Go)

## Key Features

- Multi-turn agent loop with per-instance DockerHub image (`jefzda/sweap-images:{tag}`)
- Sentinel-based patch submission protocol (`COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT`)
- Container-side evaluation: `git apply` patch, run instance's `run_script.sh`, parse with `parser.py`, then check `(fail_to_pass | pass_to_pass) ⊆ PASSED`
- Supports both `toolcall` (function-calling) and `backticks` (text-based) action protocols

## Evaluation Notes

- Requires `pip install evalscope[sandbox]` (provides Docker SDK via ms-enclave)
- Requires the `scaleapi/SWE-bench_Pro-os` repository for per-instance run scripts and Dockerfiles. By default this is auto-cloned to `~/.cache/evalscope/swe_bench_pro/SWE-bench_Pro-os` and pinned to commit `ca10a60`. To use an existing clone, set `extra_params.swe_bench_pro_repo_path`.
- Both the agent loop and the per-instance evaluation share a single sandbox configuration via `TaskConfig.sandbox.default_config` (passed straight to ms_enclave `DockerSandboxConfig`). Set `memory_limit` / `cpu_limit` there to avoid OOM-Killed test runs (e.g. NodeBB); `platform` defaults to `linux/amd64` so amd64-only sweap-images work on Apple Silicon out of the box.

See the [user guide](https://evalscope.readthedocs.io/en/latest/third_party/swe_bench_pro.html) for setup, parameters, and troubleshooting.
"""

_EXTRA_PARAMS: Dict[str, Any] = {
    'swe_bench_pro_repo_path': {
        'type': 'str',
        'description': (
            'Local path to a clone of scaleapi/SWE-bench_Pro-os. '
            'If empty, auto-cloned to ~/.cache/evalscope/swe_bench_pro/SWE-bench_Pro-os '
            'and pinned to commit ca10a60.'
        ),
        'value': '',
    },
    'dockerhub_username': {
        'type': 'str',
        'description': 'DockerHub user/org hosting the sweap-images repository.',
        'value': 'jefzda',
    },
    'action_protocol': {
        'type': 'str',
        'description': (
            'Agent action protocol: "toolcall" (function-calling) or '
            '"backticks" (text-based fallback for models without function-calling support).'
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
    'eval_timeout': {
        'type': 'int',
        'description': 'Per-instance evaluation timeout in seconds.',
        'value': 3600,
    },
}


@register_benchmark(
    BenchmarkMeta(
        name='swe_bench_pro',
        pretty_name='SWE-bench_Pro',
        tags=[Tags.CODING],
        description=_DESCRIPTION,
        dataset_id='ScaleAI/SWE-bench_Pro',
        metric_list=['acc'],
        eval_split='test',
        prompt_template='{question}',
        extra_params=_EXTRA_PARAMS,
    )
)
class SWEBenchProAgenticAdapter(AgentLoopAdapter):

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

        self.action_protocol: str = self.extra_params.get('action_protocol', 'toolcall')
        if self.action_protocol not in {'toolcall', 'backticks'}:
            raise ValueError(f'Invalid action_protocol={self.action_protocol!r}; must be "toolcall" or "backticks".')
        self.max_steps = int(self.extra_params.get('max_steps', 250))
        self.command_timeout = float(self.extra_params.get('command_timeout', 60.0))
        # Hardcoded: must match the ``cd /app`` in utils.build_entry_script.
        self.working_dir: str = '/app'
        self.eval_timeout: int = int(self.extra_params.get('eval_timeout', 3600))
        self.dockerhub_username: str = self.extra_params.get('dockerhub_username', 'jefzda')

        if is_build_doc():
            self.repo_path = ''
        else:
            from .utils import ensure_pro_repo
            self.repo_path = ensure_pro_repo(self.extra_params.get('swe_bench_pro_repo_path', ''))

    # ------------------------------------------------------------------
    # Dataset loading
    # ------------------------------------------------------------------

    def record_to_sample(self, record: Dict[str, Any]) -> Sample:
        return Sample(
            input=record['problem_statement'],
            metadata={
                'instance_id': record['instance_id'],
                'repo': record['repo'],
                'base_commit': record['base_commit'],
                'problem_statement': record['problem_statement'],
                'patch': record.get('patch', ''),
                'test_patch': record.get('test_patch', ''),
                'fail_to_pass': record.get('fail_to_pass', '[]'),
                'pass_to_pass': record.get('pass_to_pass', '[]'),
                'before_repo_set_cmd': record.get('before_repo_set_cmd', ''),
                'selected_test_files_to_run': record.get('selected_test_files_to_run', '[]'),
                'repo_language': record.get('repo_language'),
                'requirements': record.get('requirements'),
                'interface': record.get('interface'),
                'issue_specificity': record.get('issue_specificity'),
                'issue_categories': record.get('issue_categories'),
                'dockerhub_tag': record.get('dockerhub_tag', ''),
            },
        )

    def _post_process_samples(self):
        from .utils import get_dockerhub_image_uri, load_instance_resources

        build_doc = is_build_doc()
        samples = self.test_dataset[self.default_subset]
        for sample in samples:
            md = sample.metadata
            tag = md.get('dockerhub_tag') or ''
            if tag:
                md['docker_image'] = f'{self.dockerhub_username}/sweap-images:{tag}'
            else:
                md['docker_image'] = get_dockerhub_image_uri(md['instance_id'], md['repo'], self.dockerhub_username)
            if not build_doc:
                md['_resources'] = load_instance_resources(self.repo_path, md['instance_id'])
            existing_tools = list(sample.tools or [])
            if not any(t.name == 'bash' for t in existing_tools):
                existing_tools.append(BASH_TOOL_INFO)
            sample.tools = existing_tools
        super()._post_process_samples()

    # ------------------------------------------------------------------
    # Agent loop hooks
    # ------------------------------------------------------------------

    def build_strategy(self, sample: Sample) -> AgentStrategy:
        if self.action_protocol == 'toolcall':
            from evalscope.agent.strategies.swe_bench import SweBenchToolcallStrategy
            return SweBenchToolcallStrategy()
        from evalscope.agent.strategies.swe_bench import SweBenchBackticksStrategy
        return SweBenchBackticksStrategy()

    def build_tools(self, sample: Sample):
        return {'bash': run_bash}

    def _user_sandbox_config(self) -> Dict[str, Any]:
        """Read ``TaskConfig.sandbox.default_config`` as the user-tunable base.

        Both the agent loop sandbox and the per-instance eval sandbox consume
        this dict, so users configure memory_limit / cpu_limit / platform /
        network_enabled / ... in **one** place. Ignores ``sandbox.enabled`` —
        SWE-bench_Pro is always sandboxed. Defaults ``platform`` to
        ``linux/amd64`` so the amd64-only sweap-images work on Apple Silicon
        without extra configuration.
        """
        cfg: Dict[str, Any] = {}
        if self._task_config is not None and self._task_config.sandbox is not None:
            cfg = dict(self._task_config.sandbox.default_config or {})
        cfg.setdefault('platform', 'linux/amd64')
        return cfg

    def build_environment(self, sample: Sample) -> Optional[AgentEnvironment]:
        from evalscope.agent.environments.enclave import EnclaveAgentEnvironment

        image = sample.metadata.get('docker_image')
        if not image:
            raise RuntimeError(
                f"docker_image missing for instance {sample.metadata.get('instance_id')!r}; "
                'did _post_process_samples run?'
            )
        agent_required = {
            'image': image,
            'working_dir': self.working_dir,
            'pull_progress': True,
            'pull_progress_interval': 5.0,
            'environment': {
                'PAGER': 'cat',
                'MANPAGER': 'cat',
                'LESS': '-R',
                'PIP_PROGRESS_BAR': 'off',
                'TQDM_DISABLE': '1',
            },
        }
        sandbox_config = merge_sandbox_config_dicts(self._user_sandbox_config(), agent_required)
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
        """Recover the agent's patch from ``self.working_dir`` (``/app``).

        External CLI agents do not implement the
        ``COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT`` sentinel protocol, so
        the patch is reconstructed from the working tree against
        ``HEAD`` instead. The returned diff is fed straight into
        :meth:`match_score`'s ``eval_instance``, reusing the entire
        SWE-bench_Pro evaluation pipeline.
        """
        from evalscope.agent.external.helpers import extract_patch
        return await extract_patch(env, cwd=self.working_dir)

    # ------------------------------------------------------------------
    # Final answer extraction
    # ------------------------------------------------------------------

    def _extract_final_answer(self, result, strategy: AgentStrategy) -> str:
        answer = strategy.extract_final_answer(result)
        if answer:
            return answer
        # Fallback: scan the last assistant message for a unified diff.
        last_assistant = ''
        for msg in reversed(result.messages):
            if getattr(msg, 'role', '') == 'assistant':
                last_assistant = str(getattr(msg, 'content', '') or '') or getattr(msg, 'text', '') or ''
                if last_assistant:
                    break
        try:
            from swebench.inference.make_datasets.utils import extract_diff
            return extract_diff(last_assistant) or ''
        except Exception:
            # Minimal fallback: return raw text starting at the first ``diff --git`` block.
            idx = last_assistant.find('diff --git ')
            return last_assistant[idx:] if idx >= 0 else ''

    def extract_answer(self, prediction: str, task_state) -> str:
        return prediction or ''

    # ------------------------------------------------------------------
    # Scoring
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
        md = task_state.metadata
        result = eval_instance(
            metadata=md,
            pred=filtered_prediction,
            resources=md['_resources'],
            image=md['docker_image'],
            log_dir=self._task_config.work_dir,
            timeout=self.eval_timeout,
            sandbox_config=self._user_sandbox_config(),
        )
        score.value = {'acc': float(result.get('resolved', 0.0))}
        score.metadata = result
        return score
