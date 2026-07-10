from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from evalscope.agent.tools.bash import BASH_TOOL_INFO, run_bash
from evalscope.api.agent import AgentEnvironment, NativeAgentConfig
from evalscope.api.benchmark import BenchmarkMeta
from evalscope.api.benchmark.adapters import AgentLoopAdapter
from evalscope.api.dataset import DatasetDict, LocalDataLoader, Sample
from evalscope.api.evaluator import InferenceResult, TaskState
from evalscope.api.messages import ChatMessageSystem
from evalscope.api.metric import AggScore, SampleScore, Score
from evalscope.api.registry import register_benchmark
from evalscope.constants import HubType, JudgeStrategy, Tags
from evalscope.utils.import_utils import check_import
from .utils import METRIC_NAMES, TemporaryLocalAgentEnvironment, WideSearchScorer, aggregate_official_scores

DATASET_ID = 'bytedance-community/WideSearch'

SYSTEM_PROMPTS = {
    'en': """# Role
You are an expert in online search. You task is gathering relevant information using advanced online search tools based on the user's query, and providing accurate answers according to the search results.

# Task Description
Upon receiving the user's query, you must thoroughly analyze and understand the user's requirements. In order to effectively address the user's query, you should make the best use of the provided tools to acquire comprehensive and reliable information and data. Below are the principles you should adhere to while performing this task:

- Fully understand the user's needs: Analyze the user's query, if necessary, break it down into smaller components to ensure a clear understanding of the user's primary intent.
- Flexibly use tools: After fully comprehending the user's needs, employ the provided tools to retrieve the necessary information.If the information retrieved previously is deemed incomplete or inaccurate and insufficient to answer the user's query, reassess what additional information is required and invoke the tool again until all necessary data is obtained.""",  # noqa: E501
    'zh': """# 角色设定
你是一位联网信息搜索专家，你需要根据用户的问题，通过联网搜索来搜集相关信息，然后根据这些信息来回答用户的问题。

# 任务描述
当你接收到用户的问题后，你需要充分理解用户的需求，利用我提供给你的工具，获取相对应的信息、资料，以解答用户的问题。
以下是你在执行任务过程中需要遵循的原则：
- 充分理解用户需求：你需要全面分析和理解用户的问题，必要时对用户的问题进行拆解，以确保领会到用户问题的主要意图。
- 灵活使用工具：当你充分理解用户需求后，请你使用我提供的工具获取信息；当你认为上次工具获取到的信息不全或者有误，以至于不足以回答用户问题时，请思考还需要搜索什么信息，再次调用工具获取信息，直至信息完备。""",
}

EXTRA_PARAMS: Dict[str, Any] = {
    'max_steps': {
        'type': 'int',
        'description': 'Maximum number of function-calling AgentLoop steps per sample.',
        'value': 50,
    },
    'environment_type': {
        'type': 'str',
        'description': 'Environment used by the built-in bash tool.',
        'value': 'local',
        'choices': ['local', 'docker'],
    },
    'command_timeout': {
        'type': 'float',
        'description': 'Default timeout for bash commands in local or Docker environments.',
        'value': 120.0,
    },
    'docker_image': {
        'type': 'str',
        'description': 'Docker image used when environment_type is docker.',
        'value': 'python:3.11-slim',
    },
    'network_enabled': {
        'type': 'bool',
        'description': 'Allow network access in the Docker environment.',
        'value': True,
    },
}

DESCRIPTION = """
## Overview

WideSearch evaluates search agents on broad information-seeking tasks: collecting a large, complete set of atomic facts
from the live web and organizing them into one strictly structured Markdown table. The benchmark contains 200 manually
curated tasks, evenly split between English and Chinese.

## Task and Runtime

- **Input**: A natural-language collection request with an explicit Markdown table schema
- **Output**: One complete Markdown table
- **Agent**: EvalScope's benchmark-owned AgentLoop with the official single-agent prompt and function-calling strategy
- **Tools**: Bash by default; optional MCP servers are merged from ``NativeAgentConfig.mcp_servers``
- **Environment**: Per-sample temporary local directory by default, optionally a Docker sandbox

The local environment uses the host network and is not a security sandbox: absolute paths can still access host files.
Use it only with trusted models. Set ``environment_type='docker'`` for isolation, or attach search and page-fetching MCP
servers for a more representative search-agent setup. The official multi-agent ``create_sub_agents`` baseline is not
implemented by this adapter.

## Evaluation

The scorer preserves the official table alignment and hybrid cell-level evaluation pipeline. It parses the Markdown
table, aligns column names and primary-key entities with an LLM judge, then applies per-column exact, numerical, date,
URL, or semantic matching. Rule-only judging is not supported; provide ``judge_model_args`` with
``judge_strategy='auto'`` or ``'llm'``. GPT-4.1-2025-04-14 is recommended for comparison with the paper.

Each trial reports table success rate and row/item precision, recall, and F1. Repeated trials are aggregated into the
paper's Avg@N, Pass@N, and Max@N metrics for ``all``, ``en``, and ``zh`` without evaluating samples more than once.
Use ``repeats=4`` to reproduce the paper's reporting shape.

## Configuration

- ``max_steps``: 50 by default
- ``environment_type``: ``local`` (default) or ``docker``
- ``command_timeout``: 120 seconds for bash commands
- ``docker_image``: ``python:3.11-slim``
- ``network_enabled``: enabled by default for Docker

Resources: [Paper](https://arxiv.org/abs/2508.07999) |
[GitHub](https://github.com/ByteDance-Seed/WideSearch) |
[Dataset](https://modelscope.cn/datasets/bytedance-community/WideSearch)

## Usage

Default local bash evaluation:

```python
from evalscope import TaskConfig, run_task

run_task(TaskConfig(
    model='YOUR_MODEL',
    datasets=['wide_search'],
    judge_strategy='llm',
    judge_model_args={'model_id': 'YOUR_JUDGE_MODEL'},
))
```

Choose the local or Docker environment through benchmark parameters:

```python
TaskConfig(
    model='YOUR_MODEL',
    datasets=['wide_search'],
    dataset_args={'wide_search': {'extra_params': {'environment_type': 'docker'}}},
    judge_strategy='llm',
    judge_model_args={'model_id': 'YOUR_JUDGE_MODEL'},
)
```

Reproduce the paper's four-trial report shape with bash and the Fetch MCP server:

```python
import sys

from evalscope import TaskConfig, run_task
from evalscope.api.agent import NativeAgentConfig
from evalscope.api.agent.mcp import MCPServerConfigStdio

run_task(TaskConfig(
    model='YOUR_MODEL',
    datasets=['wide_search'],
    repeats=4,
    agent_config=NativeAgentConfig(mcp_servers=[MCPServerConfigStdio(
        command=sys.executable,
        args=['-m', 'mcp_server_fetch', '--ignore-robots-txt'],
        name='fetch',
    )]),
    judge_strategy='llm',
    judge_model_args={'model_id': 'YOUR_JUDGE_MODEL'},
))
```
"""


@register_benchmark(
    BenchmarkMeta(
        name='wide_search',
        pretty_name='WideSearch',
        tags=[Tags.AGENT, Tags.MULTI_TURN, Tags.RETRIEVAL],
        description=DESCRIPTION,
        dataset_id=DATASET_ID,
        subset_list=['default'],
        default_subset='default',
        eval_split='full',
        prompt_template='{question}',
        metric_list=list(METRIC_NAMES),
        extra_params=EXTRA_PARAMS,
        paper_url='https://arxiv.org/abs/2508.07999',
    )
)
class WideSearchAdapter(AgentLoopAdapter):
    """Official single-agent WideSearch benchmark adapter."""

    strategy_name = 'function_calling'
    max_steps_default = 50

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        check_import('dateparser', extra='wide_search', raise_error=True, feature_name='WideSearch evaluation')
        self._use_llm_judge = True
        self.max_steps = int(self.extra_params.get('max_steps', self.max_steps_default))
        self.environment_type = str(self.extra_params.get('environment_type', 'local'))
        self.command_timeout = float(self.extra_params.get('command_timeout', 120.0))
        self.docker_image = str(self.extra_params.get('docker_image', 'python:3.11-slim'))
        self.network_enabled = bool(self.extra_params.get('network_enabled', True))
        if self.max_steps <= 0:
            raise ValueError('WideSearch max_steps must be greater than 0.')
        if self.environment_type not in {'local', 'docker'}:
            raise ValueError("WideSearch environment_type must be 'local' or 'docker'.")
        if self.command_timeout <= 0:
            raise ValueError('WideSearch command_timeout must be greater than 0.')
        self._dataset_root: Optional[Path] = None
        self._judge_lock = threading.Lock()

    def load(self) -> Tuple[DatasetDict, None]:
        dataset_root = self._resolve_dataset_root()
        self._dataset_root = dataset_root
        data_path = dataset_root / 'widesearch.jsonl'
        if not data_path.exists():
            raise FileNotFoundError(f'WideSearch data file not found: {data_path}')
        dataset = LocalDataLoader(
            data_id_or_path=str(data_path),
            split=self.eval_split,
            subset='default',
            sample_fields=self.record_to_sample,
            limit=self.limit,
            repeats=self.repeats,
            shuffle=self.shuffle,
        ).load()
        return DatasetDict({'default': dataset}), None

    def _resolve_dataset_root(self) -> Path:
        if Path(self.dataset_id).exists():
            return Path(self.dataset_id).expanduser().resolve()
        if str(self.dataset_hub).lower() != HubType.MODELSCOPE:
            raise ValueError('WideSearch currently supports ModelScope or dataset_args.local_path.')
        from modelscope import dataset_snapshot_download
        return Path(dataset_snapshot_download(self.dataset_id))

    def record_to_sample(self, record: Dict[str, Any]) -> Sample:
        if self._dataset_root is None:
            raise RuntimeError('WideSearch dataset root is not initialized.')
        instance_id = str(record['instance_id'])
        gold_path = self._dataset_root / 'widesearch_gold' / f'{instance_id}.csv'
        if not gold_path.exists():
            raise FileNotFoundError(f'WideSearch gold file not found: {gold_path}')
        evaluation = record['evaluation']
        if isinstance(evaluation, str):
            evaluation = json.loads(evaluation)
        return Sample(
            input=str(record['query']),
            target=gold_path.read_text(encoding='utf-8'),
            tools=[BASH_TOOL_INFO],
            metadata={
                'instance_id': instance_id,
                'language': str(record['language']),
                'evaluation': evaluation,
            },
        )

    def build_tools(self, sample: Sample) -> Dict[str, Any]:
        return {'bash': run_bash}

    def _resolve_tools(self, sample: Sample, ac: Any) -> Tuple[Dict[str, Any], List[Any]]:
        handlers, tools = super()._resolve_tools(sample, ac)
        if isinstance(ac, NativeAgentConfig) and ac.command_timeout is not None:
            return handlers, tools
        from evalscope.agent.tools.bash import apply_bash_command_timeout_defaults
        return apply_bash_command_timeout_defaults(handlers, tools, self.command_timeout)

    def build_environment(self, sample: Sample) -> Optional[AgentEnvironment]:
        sample_id = sample.metadata.get('instance_id') or sample.id or 'unknown'
        if self.environment_type == 'local':
            return TemporaryLocalAgentEnvironment(sample_id=sample_id)
        check_import('ms_enclave', extra='sandbox', raise_error=True, feature_name='WideSearch Docker environment')
        from evalscope.agent.environments.enclave import EnclaveAgentEnvironment
        return EnclaveAgentEnvironment(
            engine='docker',
            sandbox_config={
                'image': self.docker_image,
                'network_enabled': self.network_enabled,
            },
            timeout=self.command_timeout,
        )

    def build_initial_messages(self, sample: Sample) -> List[Any]:
        messages = super().build_initial_messages(sample)
        language = str(sample.metadata.get('language', 'en'))
        return [ChatMessageSystem(content=SYSTEM_PROMPTS.get(language, SYSTEM_PROMPTS['en']))] + messages

    def _on_inference(self, model: Any, sample: Sample) -> InferenceResult:
        result = super()._on_inference(model, sample)
        if not self._reached_max_steps(result):
            return result
        from evalscope.api.agent import EventType
        from evalscope.api.messages import ChatMessageUser
        final_message = ChatMessageUser(
            content=
            '[Max Step] The tool has been used too many times. Please stop invoking the tool immediately and answer the user\'s question.'
        )
        final_input = list(result.messages or []) + [final_message]
        final_output = model.generate(input=final_input, tools=None)
        messages = final_input + [final_output.message]
        if result.trace is not None:
            step = result.trace.max_steps
            result.trace.add_event(
                step=step,
                type=EventType.NUDGE,
                message_id=final_message.id,
                payload={'reason': 'max_steps_finalization'},
            )
            usage = None
            if final_output.usage is not None:
                usage = {
                    'input': final_output.usage.input_tokens,
                    'output': final_output.usage.output_tokens,
                    'total': final_output.usage.total_tokens,
                }
            result.trace.add_event(
                step=step,
                type=EventType.MODEL_GENERATE,
                message_id=final_output.message.id,
                token_usage=usage,
                payload={
                    'stop_reason': final_output.stop_reason,
                    'phase': 'max_steps_finalization'
                },
            )
            if final_output.completion.strip():
                result.trace.add_event(
                    step=step,
                    type=EventType.SUBMIT,
                    message_id=final_output.message.id,
                    payload={
                        'final_answer': final_output.completion,
                        'phase': 'max_steps_finalization'
                    },
                )
                if result.trace.total_usage is not None and final_output.usage is not None:
                    result.trace.total_usage += final_output.usage
        return InferenceResult(output=final_output, messages=messages, trace=result.trace)

    @staticmethod
    def _reached_max_steps(result: InferenceResult) -> bool:
        if result.trace is None:
            return False
        from evalscope.api.agent import EventType
        return any(
            event.type == EventType.ERROR and event.payload.get('message') == 'max_steps_exceeded'
            for event in result.trace.events
        )

    def calculate_metrics(self, task_state: TaskState) -> SampleScore:
        self._validate_judge_config()
        with self._judge_lock:
            judge = self.llm_judge
        scorer = WideSearchScorer(judge=judge.judge)
        result = scorer.evaluate(
            prediction=task_state.output.completion,
            gold_csv=task_state.target,
            evaluation=task_state.metadata['evaluation'],
        )
        score = Score(
            extracted_prediction=task_state.output.completion,
            prediction=task_state.output.completion,
            value=result.values,
            explanation=result.diagnostics.get('error') or 'Official WideSearch table evaluation completed.',
            metadata=result.diagnostics,
            main_score_name='success_rate',
        )
        return SampleScore(
            score=score,
            sample_id=task_state.sample_id,
            group_id=task_state.group_id,
            sample_metadata=task_state.metadata,
        )

    def _validate_judge_config(self) -> None:
        if self.judge_strategy not in {JudgeStrategy.AUTO, JudgeStrategy.LLM}:
            raise ValueError("WideSearch requires judge_strategy='auto' or 'llm'.")
        if not self._task_config.judge_model_args:
            raise ValueError('WideSearch requires judge_model_args for official table alignment and scoring.')

    def aggregate_scores(self, sample_scores: List[SampleScore]) -> List[AggScore]:
        return aggregate_official_scores(sample_scores)
