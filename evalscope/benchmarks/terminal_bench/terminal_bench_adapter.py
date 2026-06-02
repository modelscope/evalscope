import json
import os
import shutil
import uuid
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

from evalscope.api.agent import AgentTrace, AgentTraceEvent, EventType
from evalscope.api.benchmark import AgentAdapter, BenchmarkMeta
from evalscope.api.dataset import DatasetDict, DictDataLoader, Sample
from evalscope.api.evaluator import InferenceResult
from evalscope.api.messages import ChatMessage, ChatMessageAssistant, ChatMessageTool, ChatMessageUser
from evalscope.api.metric import Score
from evalscope.api.model import Model, ModelOutput
from evalscope.api.registry import register_benchmark
from evalscope.api.tool import ToolCall, ToolFunction
from evalscope.constants import DEFAULT_EVALSCOPE_CACHE_DIR, Tags
from evalscope.utils.function_utils import AsyncioLoopRunner
from evalscope.utils.import_utils import check_import
from evalscope.utils.logger import get_logger

logger = get_logger()

COMMON_EXTRA_PARAMS = {
    'environment_type': {
        'type': 'str',
        'description': 'Environment type for running the benchmark.',
        'value': 'docker',
        'choices': ['docker', 'daytona', 'e2b', 'modal']
    },
    'agent_name': {
        'type': 'str',
        'description': 'Agent type to be used in Harbor. Only terminus-2 uses the evalscope model for inference; '
        'other agents (claude-code, codex, etc.) run as standalone CLI tools with their own API keys.',
        'value': 'terminus-2',
        'choices': [
            'oracle', 'terminus-2', 'claude-code', 'codex', 'qwen-coder', 'openhands', 'opencode', 'mini-swe-agent'
        ],
    },
    'timeout_multiplier': {
        'type': 'float',
        'description': 'Timeout multiplier. If timeout errors occur, consider increasing this value.',
        'value': 1.0,
    },
    'max_turns': {
        'type': 'int',
        'description': 'Maximum number of turns for the agent to complete the task.',
        'value': 200,
    },
    'environment_kwargs': {
        'type': 'dict',
        'description': 'Extra kwargs passed to Harbor EnvironmentConfig. '
        'Supported keys: override_cpus, override_memory_mb, override_storage_mb, override_gpus, '
        'force_build, delete, env, etc.',
        'value': {},
    },
}


def _validate_environment_requirements(environment_type: str):
    environment_type = (environment_type or '').strip().lower()
    if environment_type != 'docker':
        return

    if shutil.which('docker') is None:
        raise RuntimeError(
            'Terminal-Bench with environment_type=\'docker\' requires the Docker CLI to be installed in the '
            'environment running EvalScope. Mounting /var/run/docker.sock only exposes the Docker daemon socket; '
            'it does not provide the docker command. Install the Docker CLI in the container or switch '
            'environment_type to \'daytona\', \'e2b\', or \'modal\'.'
        )


class _TerminalBenchBase(AgentAdapter):
    """Shared logic for Terminal-Bench adapters."""

    hub_dataset_name: str = ''

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        check_import('harbor', extra='terminal_bench', raise_error=True, feature_name=self.pretty_name)
        self.environment_type = self.extra_params.get('environment_type', 'docker')
        self.agent_name = self.extra_params.get('agent_name', 'terminus-2')
        self.timeout_multiplier = self.extra_params.get('timeout_multiplier', 1.0)
        self.max_turns = self.extra_params.get('max_turns', 200)
        self.environment_kwargs = self.extra_params.get('environment_kwargs', {})

    def load(self):
        _validate_environment_requirements(self.environment_type)

        from harbor.models.job.config import DatasetConfig

        config = DatasetConfig(
            name=self.hub_dataset_name,
            overwrite=self.force_redownload,
            download_dir=Path(os.path.join(DEFAULT_EVALSCOPE_CACHE_DIR, self.name)),
        )

        logger.info(f'Downloading dataset for {self.pretty_name} from Harbor Hub...')
        task_configs = AsyncioLoopRunner.run(config.get_task_configs())

        datasets = {}
        dataset = DictDataLoader(
            dict_list=[tc.model_dump(mode='json') for tc in task_configs],
            limit=self.limit,
            repeats=self.repeats,
            sample_fields=self.record_to_sample,
            shuffle=self.shuffle,
        ).load()

        datasets[self.eval_split] = dataset

        test_dataset = DatasetDict(datasets)
        return test_dataset, None

    def record_to_sample(self, record) -> Sample:
        return Sample(input='', metadata=record)

    def _on_inference(self, model: Model, sample: Sample) -> InferenceResult:
        from harbor.models.trial.config import AgentConfig, EnvironmentConfig
        from harbor.models.trial.config import TaskConfig as TrialTaskConfig
        from harbor.models.trial.config import TrialConfig
        from harbor.trial.trial import Trial

        from .utils import HarborLLM

        env_kwargs = {k: v for k, v in self.environment_kwargs.items() if k != 'type'}
        environment_config = EnvironmentConfig(type=self.environment_type, **env_kwargs)

        agent_kwargs = {'max_turns': self.max_turns}
        if self.agent_name == 'terminus-2':
            agent_kwargs.update({
                'parser_name': 'json',
                'enable_summarize': True,
                'proactive_summarization_threshold': 8000,
                'collect_rollout_details': False,
            })

        agent_config = AgentConfig(
            name=self.agent_name,
            model_name=model.name,
            kwargs=agent_kwargs,
        )

        trial_task_config = TrialTaskConfig.model_validate(sample.metadata)
        trial_config = TrialConfig(
            task=trial_task_config,
            trials_dir=Path(self.output_dir) / 'trials',
            agent=agent_config,
            environment=environment_config,
            timeout_multiplier=self.timeout_multiplier,
        )

        try:

            async def _run_trial():
                trial = await Trial.create(trial_config)
                if self.agent_name == 'terminus-2':
                    trial.agent._llm = HarborLLM(model=model)
                return await trial.run()

            result = AsyncioLoopRunner.run(_run_trial())
        except Exception as e:
            if hasattr(e, 'exceptions'):
                for i, sub_exc in enumerate(e.exceptions):
                    logger.warning(f'--- Sub-exception {i + 1} ---')
                    logger.warning(sub_exc)
            else:
                logger.warning(e)
            raise e

        result_dict = result.model_dump(mode='json')
        sample.metadata['result'] = result_dict

        output = ModelOutput.from_content(
            model=model.name,
            content=result_dict.get('trial_uri', ''),
        )
        trace, messages = self._load_harbor_trace(result_dict)
        return InferenceResult(output=output, trace=trace, messages=messages)

    def _load_harbor_trace(self, result_dict: dict) -> Tuple[Optional[AgentTrace], Optional[List[ChatMessage]]]:
        trial_uri = result_dict.get('trial_uri') or ''
        if trial_uri.startswith('file://'):
            trajectory_path = Path(trial_uri[7:]) / 'agent' / 'trajectory.json'
        else:
            return None, None
        if not trajectory_path.exists():
            return None, None
        try:
            raw = json.loads(trajectory_path.read_text())
        except Exception:
            return None, None

        agent_info = raw.get('agent', {})
        model_name = agent_info.get('model_name')
        trace = AgentTrace(
            framework=agent_info.get('name', 'harbor'),
            environment=self.environment_type,
        )
        messages: List[ChatMessage] = []
        prev_ts: Optional[float] = None

        for step in raw.get('steps', []):
            msg_id = uuid.uuid4().hex[:8]
            source = step.get('source', '')
            content = step.get('message', '')
            step_id = step.get('step_id', 0)

            ts_epoch: Optional[float] = None
            ts_raw = step.get('timestamp')
            if ts_raw:
                try:
                    ts_epoch = datetime.fromisoformat(ts_raw).timestamp()
                except (ValueError, TypeError):
                    pass

            if source == 'agent':
                latency_ms = round((ts_epoch - prev_ts) * 1000) if ts_epoch and prev_ts else None

                tool_calls = []
                for tc in step.get('tool_calls', []):
                    tool_calls.append(
                        ToolCall(
                            id=tc.get('tool_call_id',
                                      uuid.uuid4().hex[:8]),
                            function=ToolFunction(
                                name=tc.get('function_name', 'bash_command'),
                                arguments=tc.get('arguments', {}),
                            ),
                        )
                    )
                messages.append(
                    ChatMessageAssistant(
                        id=msg_id,
                        content=content,
                        model=model_name,
                        tool_calls=tool_calls or None,
                    )
                )
                token_usage = None
                step_metrics = step.get('metrics')
                if step_metrics:
                    token_usage = {
                        'input': step_metrics.get('prompt_tokens', 0),
                        'output': step_metrics.get('completion_tokens', 0),
                    }
                trace.add(
                    AgentTraceEvent(
                        step=step_id,
                        type=EventType.MODEL_GENERATE,
                        message_id=msg_id,
                        timestamp=ts_epoch or 0,
                        latency_ms=latency_ms,
                        token_usage=token_usage,
                    )
                )

                for tc in tool_calls:
                    trace.add(
                        AgentTraceEvent(
                            step=step_id,
                            type=EventType.TOOL_CALL,
                            timestamp=ts_epoch or 0,
                            payload={
                                'id': tc.id,
                                'name': tc.function.name,
                                'arguments': tc.function.arguments
                            },
                        )
                    )

                obs = step.get('observation', {})
                results = obs.get('results', [])
                for i, result in enumerate(results):
                    tool_msg_id = uuid.uuid4().hex[:8]
                    # Harbor batches N keystrokes per step but returns one
                    # combined terminal observation.  Link the (single) result
                    # to the last tool_call so the UI shows it after all calls.
                    link_idx = len(tool_calls) - 1 if len(results) == 1 and tool_calls else i
                    call_id = tool_calls[link_idx].id if link_idx < len(tool_calls) else None
                    fn_name = tool_calls[link_idx].function.name if link_idx < len(tool_calls) else 'bash_command'
                    messages.append(
                        ChatMessageTool(
                            id=tool_msg_id,
                            content=result.get('content', ''),
                            tool_call_id=call_id,
                            function=fn_name,
                        )
                    )
                    trace.add(
                        AgentTraceEvent(
                            step=step_id,
                            type=EventType.TOOL_RESULT,
                            message_id=tool_msg_id,
                            timestamp=ts_epoch or 0,
                            payload={'id': call_id},
                        )
                    )
            else:
                messages.append(ChatMessageUser(id=msg_id, content=content))
                trace.add(
                    AgentTraceEvent(
                        step=step_id,
                        type=EventType.ENV_EXEC,
                        message_id=msg_id,
                        timestamp=ts_epoch or 0,
                        payload={'source': source},
                    )
                )

            if ts_epoch:
                prev_ts = ts_epoch

        metrics = raw.get('final_metrics', {})
        if metrics:
            trace.add(
                AgentTraceEvent(
                    step=len(raw.get('steps', [])),
                    type=EventType.SUBMIT,
                    token_usage={
                        'input': metrics.get('total_prompt_tokens', 0),
                        'output': metrics.get('total_completion_tokens', 0),
                        'cached': metrics.get('total_cached_tokens', 0),
                    },
                )
            )
        return trace, messages

    def match_score(self, original_prediction, filtered_prediction, reference, task_state):
        result = task_state.metadata.get('result', {})
        try:
            reward = result.get('verifier_result', {}).get('rewards', {}).get('reward', 0)
        except Exception:
            reward = 0
        score = Score(
            extracted_prediction=filtered_prediction,
            prediction=original_prediction,
            value={'acc': reward},
            metadata=result,
        )
        return score


@register_benchmark(
    BenchmarkMeta(
        name='terminal_bench_v2',
        pretty_name='Terminal-Bench-2.0',
        tags=[Tags.CODING],
        description="""
## Overview

Terminal-Bench v2 is a command-line benchmark suite that evaluates AI agents on 89 real-world, multi-step terminal tasks. Tasks range from compiling and debugging to system administration, running within isolated containers with rigorous validation.

## Task Description

- **Task Type**: Command-Line Agent Evaluation
- **Input**: Terminal task specification
- **Output**: Task completion via agent actions
- **Domains**: System administration, compilation, debugging, file operations

## Key Features

- 89 real-world terminal tasks
- Multi-step task completion requirements
- Isolated container execution environment
- Binary scoring (0/1) with auto-validation
- Multiple agent types supported (terminus-2, claude-code, codex, etc.)

## Evaluation Notes

- Requires **Python>=3.12** and `pip install evalscope[terminal_bench]`
- Environment options: docker, daytona, e2b, modal
- Configurable agent types and timeout settings
- Maximum turns configurable (default: 200)
- [Usage Example](https://evalscope.readthedocs.io/en/latest/third_party/terminal_bench.html)
""",
        dataset_id='https://hub.harborframework.com/datasets/terminal-bench/terminal-bench-2/latest',
        metric_list=['acc'],
        eval_split='test',
        prompt_template='{question}',
        extra_params=COMMON_EXTRA_PARAMS,
    )
)
class TerminalBenchV2Adapter(_TerminalBenchBase):

    hub_dataset_name = 'terminal-bench/terminal-bench-2'


@register_benchmark(
    BenchmarkMeta(
        name='terminal_bench_v2_1',
        pretty_name='Terminal-Bench-2.1',
        tags=[Tags.CODING],
        description="""
## Overview

Terminal-Bench v2.1 is an improved iteration of Terminal-Bench 2.0, with 26 task fixes addressing bugs, timeout adjustments, and reward hacking prevention. Recommended over v2.0 for new evaluations.

## Task Description

- **Task Type**: Command-Line Agent Evaluation
- **Input**: Terminal task specification
- **Output**: Task completion via agent actions
- **Domains**: System administration, compilation, debugging, file operations

## Key Features

- Verified iteration of Terminal-Bench 2.0 with 26 task fixes
- Improved robustness against reward hacking
- Isolated container execution environment
- Binary scoring (0/1) with auto-validation
- Multiple agent types supported (terminus-2, claude-code, codex, etc.)

## Evaluation Notes

- Requires **Python>=3.12** and `pip install evalscope[terminal_bench]`
- Environment options: docker, daytona, e2b, modal
- Configurable agent types and timeout settings
- Maximum turns configurable (default: 200)
- [Usage Example](https://evalscope.readthedocs.io/en/latest/third_party/terminal_bench.html)
""",
        dataset_id='https://hub.harborframework.com/datasets/terminal-bench/terminal-bench-2-1/latest',
        metric_list=['acc'],
        eval_split='test',
        prompt_template='{question}',
        extra_params=COMMON_EXTRA_PARAMS,
    )
)
class TerminalBenchV2_1Adapter(_TerminalBenchBase):

    hub_dataset_name = 'terminal-bench/terminal-bench-2-1'
