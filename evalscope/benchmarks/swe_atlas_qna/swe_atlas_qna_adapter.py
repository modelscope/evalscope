"""SWE-Atlas QnA adapter.

Drives a read-only AgentLoop per sample inside the task's own image
(``ghcr.io/scaleapi/swe-atlas:...``). The model explores the repository at
``/app`` with a ``bash`` tool and answers in prose; an LLM judge then scores
the answer against each expert-written rubric item independently.

A task counts as resolved only when every rubric item passes, which is the
leaderboard's Task Resolve Rate. Answers that came with source-code edits are
failed outright, mirroring the official programmatic check.
"""

from __future__ import annotations

from math import isfinite
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

from evalscope.agent.environments.enclave import EnclaveAgentEnvironment
from evalscope.agent.tools.bash import BASH_TOOL_INFO, run_bash
from evalscope.api.agent import AgentEnvironment, ToolExecutionOutput
from evalscope.api.benchmark import BenchmarkMeta
from evalscope.api.benchmark.adapters import AgentLoopAdapter
from evalscope.api.dataset import Sample
from evalscope.api.judge import (
    JudgeCase,
    JudgeContext,
    JudgeDefinition,
    JudgeRequest,
    OutputContract,
    ReducedVerdict,
)
from evalscope.api.messages import ChatMessageSystem, ChatMessageUser
from evalscope.api.metric import AggScore, SampleScore
from evalscope.api.registry import register_benchmark
from evalscope.api.sandbox import merge_sandbox_config_dicts, task_sandbox_env_kwargs
from evalscope.api.tool import ToolCall
from evalscope.constants import ScoringPolicy, Tags
from evalscope.utils.logger import get_logger

from .metadata import DATASET_ID, DESCRIPTION, EXTRA_PARAMS, HF_DATASET_ID, WORKING_DIR
from .prompts import (
    AGENT_INSTANCE_TEMPLATE,
    AGENT_SYSTEM_TEMPLATE,
    JUDGE_SYSTEM_PROMPT,
    JUDGE_USER_PROMPT_TEMPLATE,
)
from .utils import (
    dirty_check_command,
    extract_final_answer,
    judge_rubric_statement,
    parse_rubric,
    reset_repo_command,
    rubric_item_passed,
)

if TYPE_CHECKING:
    from evalscope.api.agent import AgentStrategy

logger = get_logger()

# The task images initialise their language runtimes from shell startup files,
# so commands run through a login shell as mini-swe-agent does.
_INTERPRETER: tuple[str, ...] = ('bash', '-lc')

_SANDBOX_STATE_KEY = '_sandbox_state'


class RubricVerdict(BaseModel):
    """One rating in the official judge response."""

    model_config = ConfigDict(extra='forbid')

    rubric_statement: str
    status: Literal['YES', 'NO']
    score: Literal['0', '1']
    justification: str

    @model_validator(mode='after')
    def validate_consistent_status_score(self) -> 'RubricVerdict':
        if (self.status == 'YES') != (self.score == '1'):
            raise ValueError('status and score must agree')
        return self


class RubricResponse(BaseModel):
    """The official single-criterion response envelope."""

    model_config = ConfigDict(extra='forbid')

    ratings: List[RubricVerdict] = Field(min_length=1, max_length=1)


# The official system prompt already dictates this JSON shape, so no extra
# instruction() is appended to it.
RUBRIC_CONTRACT = OutputContract(schema_model=RubricResponse)


class SweAtlasSandboxState(BaseModel):
    """Repository setup and cleanliness observed before sandbox teardown."""

    model_config = ConfigDict(extra='forbid')

    setup_returncode: Optional[int] = None
    setup_error: Optional[str] = None
    check_returncode: Optional[int] = None
    check_error: Optional[str] = None
    tree_status: Optional[str] = None


class SweAtlasSandboxMixin:
    """Adds repository reset and a dirty-tree check around the sandbox.

    Reset runs lazily on the first command because the environment is built
    synchronously while the sandbox only starts on first use. The dirty check
    runs in :meth:`close` because the agent loop owns the environment and tears
    it down before scoring; the public state object is shared with the sample's
    metadata so scoring can still read it.
    """

    def __init__(
        self,
        *,
        setup_command: Optional[str] = None,
        check_command: Optional[str] = None,
        state: Optional[SweAtlasSandboxState] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._setup_command = setup_command
        self._check_command = check_command
        self.state = state if state is not None else SweAtlasSandboxState()
        self._setup_done = False

    async def exec(self, cmd, **kwargs):  # type: ignore[no-untyped-def]
        if not self._setup_done:
            self._setup_done = True
            if self._setup_command:
                try:
                    result = await super().exec(['bash', '-c', self._setup_command])
                except Exception as exc:
                    self.state.setup_error = str(exc)
                    raise
                self.state.setup_returncode = result.returncode
                if result.returncode != 0:
                    detail = (result.stderr or result.stdout or '').strip()[:2000]
                    self.state.setup_error = detail or f'reset exited with code {result.returncode}'
                    logger.warning('SWE-Atlas QnA repo reset failed: %s', self.state.setup_error)
                    raise RuntimeError(self.state.setup_error)
        return await super().exec(cmd, **kwargs)

    async def close(self) -> None:
        should_check = (
            self._check_command
            and self._setup_done
            and self.state.setup_error is None
            and self.state.check_returncode is None
            and self.state.check_error is None
        )
        if should_check:
            try:
                result = await super().exec(['bash', '-c', self._check_command])
                self.state.check_returncode = result.returncode
                if result.returncode != 0:
                    detail = (result.stderr or result.stdout or '').strip()[:2000]
                    self.state.check_error = detail or f'dirty-tree check exited with code {result.returncode}'
                    logger.warning('SWE-Atlas QnA dirty-tree check failed: %s', self.state.check_error)
                else:
                    self.state.tree_status = (result.stdout or '').strip()
            except Exception as e:  # noqa: BLE001 - teardown must not mask the run
                self.state.check_error = str(e)
                logger.warning('SWE-Atlas QnA dirty-tree check failed: %s', e)
        await super().close()


class SweAtlasEnclaveEnvironment(SweAtlasSandboxMixin, EnclaveAgentEnvironment):
    """EvalScope's ms-enclave environment with SWE-Atlas repository checks."""


@register_benchmark(
    BenchmarkMeta(
        name='swe_atlas_qna',
        pretty_name='SWE-Atlas-QnA',
        tags=[Tags.CODING, Tags.AGENT, Tags.MULTI_TURN],
        description=DESCRIPTION,
        dataset_id=DATASET_ID,
        hf_dataset_id=HF_DATASET_ID,
        subset_list=['default'],
        default_subset='default',
        eval_split='test',
        prompt_template='{question}',
        metric_list=['resolve_rate', 'rubric_pass_rate'],
        primary_metric='resolve_rate',
        extra_params=EXTRA_PARAMS,
        evaluation_version='v1.1',
    )
)
class SweAtlasQnAAdapter(AgentLoopAdapter):
    strategy_name = 'function_calling'
    max_steps_default = 250
    # Rubric grading is the only scoring path; a rule-based fallback would emit
    # nothing but zeros, so `auto` must resolve to the judge.
    scoring_policy = ScoringPolicy.JUDGE_ONLY

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.working_dir: str = self.extra_params.get('working_dir', WORKING_DIR)
        self.reset_to_base_commit: bool = bool(self.extra_params.get('reset_to_base_commit', True))
        self.fail_on_source_modification: bool = bool(self.extra_params.get('fail_on_source_modification', True))
        self.command_timeout: float = float(self.extra_params.get('command_timeout', 900.0))
        if not isfinite(self.command_timeout) or self.command_timeout <= 0:
            raise ValueError('command_timeout must be a finite positive number')
        self.command_timeout_default = self.command_timeout
        self.pass_threshold: float = float(self.extra_params.get('pass_threshold', 1.0))
        if not isfinite(self.pass_threshold) or not 0.0 <= self.pass_threshold <= 1.0:
            raise ValueError('pass_threshold must be a finite number in [0, 1]')

    # ------------------------------------------------------------------
    # Dataset loading
    # ------------------------------------------------------------------

    def record_to_sample(self, record: Dict[str, Any]) -> Sample:
        rubric = parse_rubric(record.get('rubric'))
        return Sample(
            input=record['prompt'],
            target=record.get('reference_answer', ''),
            metadata={
                'task_id': record['task_id'],
                'prompt': record['prompt'],
                'reference_answer': record.get('reference_answer', ''),
                'repository_url': record.get('repository_url', ''),
                'repository_base_commit': record.get('repository_base_commit', ''),
                'language': record.get('language', ''),
                'category': record.get('category', ''),
                'docker_image': record['docker_image'],
                'rubric': rubric,
            },
        )

    def _post_process_samples(self):
        for sample in self.test_dataset[self.default_subset]:
            tools = list(sample.tools or [])
            if not any(tool.name == 'bash' for tool in tools):
                tools.append(BASH_TOOL_INFO)
            sample.tools = tools
        super()._post_process_samples()

    # ------------------------------------------------------------------
    # Agent loop hooks
    # ------------------------------------------------------------------

    def build_tools(self, sample: Sample) -> Dict[str, Any]:
        async def bash(call: ToolCall, env: Optional[AgentEnvironment]) -> Any:
            try:
                return await run_bash(call, env)
            except Exception as exc:
                if isinstance(env, SweAtlasSandboxMixin) and env.state.setup_error:
                    return ToolExecutionOutput(text=str(exc), terminate=True, final_answer='')
                raise

        return {'bash': bash}

    def _user_sandbox_config(self) -> Dict[str, Any]:
        """User sandbox defaults plus the amd64 default the task images need."""
        config = self._task_sandbox_config()
        config.setdefault('platform', 'linux/amd64')
        return config

    def build_environment(self, sample: Sample) -> Optional[AgentEnvironment]:
        metadata = sample.metadata
        image = metadata.get('docker_image')
        if not image:
            raise RuntimeError(f'docker_image missing for task {metadata.get("task_id")!r}')

        base_commit = metadata.get('repository_base_commit') or ''
        if self.reset_to_base_commit and not base_commit:
            raise RuntimeError(f'repository_base_commit missing for task {metadata.get("task_id")!r}')
        setup_command = reset_repo_command(self.working_dir, base_commit) if self.reset_to_base_commit else None
        check_command = dirty_check_command(self.working_dir) if self.fail_on_source_modification else None

        # TaskState exposes sample metadata by reference, so scoring can read
        # the state after the agent loop has closed the sandbox.
        state = SweAtlasSandboxState()
        metadata[_SANDBOX_STATE_KEY] = state

        agent_required = {
            'image': image,
            'working_dir': self.working_dir,
            'pull_progress': True,
            'pull_progress_interval': 5.0,
            'env_vars': {
                'PAGER': 'cat',
                'MANPAGER': 'cat',
                'LESS': '-R',
                'PIP_PROGRESS_BAR': 'off',
                'TQDM_DISABLE': '1',
                'GIT_PAGER': 'cat',
            },
        }
        sandbox_config = merge_sandbox_config_dicts(self._user_sandbox_config(), agent_required)

        return SweAtlasEnclaveEnvironment(
            setup_command=setup_command,
            check_command=check_command,
            state=state,
            **task_sandbox_env_kwargs(self._task_config),
            sandbox_config=sandbox_config,
            timeout=self._native_command_timeout() or self.command_timeout,
            interpreter=_INTERPRETER,
        )

    def build_initial_messages(self, sample: Sample) -> List[Any]:
        metadata = sample.metadata
        question = metadata.get('prompt') or (sample.input if isinstance(sample.input, str) else '')
        # The upstream templates are Jinja, so placeholders are substituted
        # directly instead of through str.format, which would choke on the
        # JSON braces elsewhere in the text.
        instance = AGENT_INSTANCE_TEMPLATE.replace('{{working_dir}}', self.working_dir).replace(
            '{{problem_statement}}', question
        )
        system = AGENT_SYSTEM_TEMPLATE.strip()
        messages: List[Any] = []
        if system:
            messages.append(ChatMessageSystem(content=system))
        messages.append(ChatMessageUser(content=instance))
        return messages

    # ------------------------------------------------------------------
    # Final answer extraction
    # ------------------------------------------------------------------

    def _extract_final_answer(self, result, strategy: AgentStrategy) -> str:
        answer = strategy.extract_final_answer(result)
        return extract_final_answer(answer or '')

    def extract_answer(self, prediction: str, task_state) -> str:
        return extract_final_answer(prediction or '')

    async def _external_extract_prediction(self, env, run_result, sample: Sample) -> str:
        """Recover the final assistant text captured by the external bridge."""
        return extract_final_answer(getattr(run_result, 'final_answer', '') or '')

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def judge_definition(self, context: JudgeContext) -> JudgeDefinition:
        metadata = context.task_state.metadata
        rubric = metadata.get('rubric') or []
        question = metadata.get('prompt') or ''
        sandbox_state = metadata.get(_SANDBOX_STATE_KEY)
        if not isinstance(sandbox_state, SweAtlasSandboxState):
            sandbox_state = SweAtlasSandboxState.model_validate(sandbox_state or {})
        environment_error = sandbox_state.setup_error
        setup_returncode = sandbox_state.setup_returncode
        if not environment_error and setup_returncode not in (None, 0):
            environment_error = f'repository reset exited with code {setup_returncode}'
        if self.reset_to_base_commit and not environment_error and setup_returncode is None:
            environment_error = 'repository reset did not complete'
        if self.fail_on_source_modification and not environment_error:
            environment_error = sandbox_state.check_error
            check_returncode = sandbox_state.check_returncode
            if not environment_error and check_returncode not in (None, 0):
                environment_error = f'dirty-tree check exited with code {check_returncode}'
            if not environment_error and sandbox_state.tree_status is None:
                environment_error = 'dirty-tree check did not complete'

        data_error = 'rubric must contain at least one item' if not rubric else None
        exclusion_reason = environment_error or data_error

        cases = [
            JudgeCase(
                case_id=item['id'],
                output_contract=RUBRIC_CONTRACT,
                metadata={'title': item['title'], 'negated': item['negated']},
            )
            for item in (rubric if not exclusion_reason else [])
        ]

        def request(case, placement, completed_cases, judge_context) -> JudgeRequest:
            user = JUDGE_USER_PROMPT_TEMPLATE.format(
                problem_statement=question,
                model_answer=judge_context.filtered_prediction,
                title=judge_rubric_statement(case.metadata['title']),
            )
            return JudgeRequest(
                messages=[
                    ChatMessageSystem(content=JUDGE_SYSTEM_PROMPT),
                    ChatMessageUser(content=user),
                ]
            )

        def reduce(case_verdicts, judge_context) -> ReducedVerdict:
            passed = 0
            failed_items: List[str] = []
            for verdict in case_verdicts:
                claim_met = verdict.value.ratings[0].status == 'YES'
                negated = bool((verdict.metadata or {}).get('negated'))
                if rubric_item_passed(claim_met, negated):
                    passed += 1
                else:
                    failed_items.append(verdict.case_id)

            total = len(case_verdicts)
            rubric_pass_rate = passed / total if total else 0.0
            resolved = 1.0 if total and rubric_pass_rate >= self.pass_threshold else 0.0

            tree_status = (sandbox_state.tree_status or '').strip()
            modified_source = bool(tree_status)
            if self.fail_on_source_modification and modified_source:
                # The official harness fails a task whose answer arrived with
                # source edits, however well it scored against the rubric.
                resolved = 0.0

            return ReducedVerdict(
                value={} if exclusion_reason else {'resolved': resolved, 'rubric_pass_rate': rubric_pass_rate},
                metadata={
                    'total_rubric_items': total,
                    'passed_rubric_items': passed,
                    'failed_rubric_item_ids': failed_items,
                    'pass_threshold': self.pass_threshold,
                    'modified_source': modified_source,
                    'tree_status': tree_status[:2000],
                    'setup_returncode': sandbox_state.setup_returncode,
                    'setup_error': sandbox_state.setup_error,
                    'check_returncode': sandbox_state.check_returncode,
                    'check_error': sandbox_state.check_error,
                    'environment_error': environment_error,
                    'data_error': data_error,
                },
            )

        def finalize(score, review, judge_context):
            if exclusion_reason:
                review.error = exclusion_reason
                score.explanation = exclusion_reason
                score.metadata = {**(score.metadata or {}), 'excluded_reason': exclusion_reason}
            return score

        return JudgeDefinition.workflow(
            cases=cases, request=request, reduce=reduce, main_score_name='resolved', finalize=finalize
        )

    def aggregate_scores(self, sample_scores: List[SampleScore]) -> List[AggScore]:
        # A sample whose judge review was unusable carries an empty value dict;
        # it is dropped from the means rather than counted as a failure.
        scored = [sample_score for sample_score in sample_scores if sample_score.score.value]
        if not scored:
            return []

        resolved = [float(s.score.value.get('resolved', 0.0)) for s in scored]
        rubric_rates = [float(s.score.value.get('rubric_pass_rate', 0.0)) for s in scored]
        sample_ids = [s.sample_id for s in scored]

        return [
            AggScore(
                metric_name='resolve_rate',
                score=sum(resolved) / len(resolved),
                aggregation='mean',
                num=len(resolved),
                ids=sample_ids,
                metadata={'pass_threshold': self.pass_threshold},
            ),
            AggScore(
                metric_name='rubric_pass_rate',
                score=sum(rubric_rates) / len(rubric_rates),
                aggregation='mean',
                num=len(rubric_rates),
                ids=sample_ids,
            ),
        ]
