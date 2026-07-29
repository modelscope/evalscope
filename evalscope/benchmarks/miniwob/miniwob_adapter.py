"""MiniWoB benchmark evaluated through the OpenEnv v0.4.1 protocol."""

from __future__ import annotations

import numpy as np
import os
import threading
import time
from pathlib import Path
from PIL import Image
from typing import Any, Dict, List, Optional

from evalscope.agent.strategies.function_calling import FunctionCallingStrategy
from evalscope.api.agent import (
    AgentContext,
    AgentLoopResult,
    AgentTrace,
    AgentTraceEvent,
    EventType,
    NativeAgentConfig,
    ParsedAction,
    ToolExecutionOutput,
)
from evalscope.api.benchmark import BenchmarkMeta
from evalscope.api.benchmark.adapters import AgentLoopAdapter
from evalscope.api.dataset import DatasetDict, Sample, build_dataset_from_records
from evalscope.api.environment import EnvironmentRuntimeLease, TaskEnvironmentConfig, TaskEnvironmentSession
from evalscope.api.evaluator import InferenceResult, TaskState
from evalscope.api.messages import ChatMessageUser, ContentImage, ContentText
from evalscope.api.metric import AggScore, Score
from evalscope.api.model import Model, ModelOutput
from evalscope.api.registry import get_environment_runtime, get_task_environment, register_benchmark
from evalscope.api.sandbox import DockerImageSpec, prepare_docker_image
from evalscope.api.tool import ToolCall, ToolInfo
from evalscope.api.tool.tool_info import ToolParams
from evalscope.constants import Tags
from evalscope.report import Report
from evalscope.utils.asyncio_runtime import AsyncioLoopRunner
from evalscope.utils.json_schema import JSONSchema
from evalscope.utils.logger import get_logger
from .utils import (
    BROWSERGYM_COMMIT,
    BROWSERGYM_METADATA_SHA256,
    BROWSERGYM_VERSION,
    MINIWOB_MAX_STEPS,
    MINIWOB_PROFILE,
    load_miniwob_records,
    validate_browser_action,
)

logger = get_logger()

OPENENV_VERSION = '0.4.1'
OPENENV_COMMIT = '65c506ef94bb1f7279cb4359673b3ef81031d01f'
OPENENV_PATCH_SHA256 = '465b23aaf7b3b2cadd681495d694a7dad5ca1b36be0cfb5ce5780b94ac354668'
RUNTIME_PIP_INDEX_URL = 'https://pypi.org/simple'
MINIWOB_COMMIT = '7fd85d71a4b60325c6585396ec4f48377d049838'
_EPISODE_SEMAPHORE = threading.BoundedSemaphore(4)
_IMAGE_BUILD_LOCK = threading.Lock()
_RUNTIME_IMAGE_TAG: Optional[str] = None
_PROFILE_WARNING_LOCK = threading.Lock()
_PROFILE_WARNING_EMITTED = False

_MINIWOB_ACTIONS = (
    'noop(wait_ms=1000), mouse_move(x, y), mouse_click(x, y, button="left"), '
    'mouse_dblclick(x, y, button="left"), mouse_down(x, y, button="left"), '
    'mouse_up(x, y, button="left"), scroll(delta_x, delta_y), click(bid, button="left"), '
    'keyboard_press(key), keyboard_type(text), fill(bid, value)'
)

_DESCRIPTION = """
## Overview

MiniWoB evaluates browser agents on short interactive tasks such as clicking, form filling, drag-and-drop, and
navigation. EvalScope owns the episode schedule, model loop, scoring, traces, and reports. A pinned OpenEnv v0.4.1
BrowserGym service owns the environment lifecycle and reset/step/reward protocol.

## Evaluation

- The schedule contains 625 procedural episodes: 125 BrowserGym 0.14.3 tasks and five deterministic seeds per task.
- The task catalog is downloaded once from a pinned BrowserGym GitHub commit, checksum-verified, and cached locally.
  No ModelScope or Hugging Face dataset is used.
- The primary metric is `success_rate`; `error_rate` separately reports OpenEnv runtime failures.
- Every episode uses a fixed 20-step action budget.
- `agent_config.task_environment.observation_mode` controls the observation representation. Its default is
  `axtree_screenshot`: every reset and step supplies both the accessibility tree and a PNG screenshot. Use `axtree`
  only when a text-only diagnostic run is explicitly desired.
- Screenshot mode requires a model that accepts image input and supports function calling. A text-only model may reject
  the request, ignore the image, or act using only the incomplete accessibility tree; such scores are not representative
  of the default multimodal profile.

## Action and runtime profile

The local runtime applies an EvalScope-owned, checksum-pinned patch to OpenEnv v0.4.1 so that BrowserGym uses its
official `miniwob_all` action configuration and preserves each MiniWoB task's native viewport and timeout instead of
overriding them with OpenEnv server defaults. BrowserGym itself is not forked or modified. Reports record the OpenEnv
source commit and patch checksum.

The action configuration matches BrowserGym 0.14.3, but the EvalScope profile uses a 20-step budget instead of
BrowserGym Experiments' official 10-step budget. Reports therefore set `official_browsergym_action_config=true` and
`official_browsergym_evaluation_protocol=false`; scores must not be compared directly with the official leaderboard.

## Requirements

Install with `pip install 'evalscope[miniwob]'`.
MiniWoB currently supports only the local `ms_enclave_docker` runtime, which requires Docker and builds the patched
image from a pinned OpenEnv GitHub commit on first use.
Set `EVALSCOPE_PIP_INDEX_URL` before evaluation to use a custom Python package index while building the image.
`eval_batch_size=4` is the recommended maximum concurrency.

Local mode:

```python
TaskConfig(model='qwen3-vl-plus', datasets=['miniwob'], eval_batch_size=4)
```
"""

BROWSER_ACTION_TOOL_INFO = ToolInfo(
    name='browser_action',
    description=(
        'This is the only browser tool. Always call the tool named browser_action; never call click, fill, press, '
        'or another BrowserGym action as a tool name. Put exactly one OpenEnv BrowserGym action expression in the '
        f'action string. Supported signatures: {_MINIWOB_ACTIONS}. IMPORTANT: click accepts a string BID only, such as '
        'click("13"); never pass coordinates to click. For a visual target without a BID, use mouse_click(x, y). '
        'Coordinates must be absolute pixels in the supplied screenshot, not normalized 0-1000 coordinates. The '
        'observation states the exact screenshot width and height. Examples: mouse_click(420, 260), fill("7", "text"), '
        'keyboard_press("ENTER"), or scroll(0, 300).'
    ),
    parameters=ToolParams(
        properties={
            'action': JSONSchema(
                type='string',
                description='Exactly one BrowserGym function-call expression.',
            ),
        },
        required=['action'],
    ),
)


class _MiniWobReport(Report):
    """EvalScope report carrying the fixed MiniWoB compatibility profile."""

    metadata: Dict[str, Any]


class _MiniWobStrategy(FunctionCallingStrategy):
    """Function-calling strategy restricted to one browser action per turn."""

    name = 'miniwob_openenv_function_calling'

    def parse_output(self, output: ModelOutput, ctx: AgentContext) -> ParsedAction:
        parsed = super().parse_output(output, ctx)
        if not parsed.tool_calls:
            return parsed
        if len(parsed.tool_calls) != 1 or parsed.tool_calls[0].function.name != 'browser_action':
            return ParsedAction(
                error='Call exactly one browser_action tool per turn.',
                raw_text=parsed.raw_text,
            )
        return parsed

    def tools(self, ctx: AgentContext) -> List[ToolInfo]:
        return list(ctx.tools)


@register_benchmark(
    BenchmarkMeta(
        name='miniwob',
        pretty_name='MiniWoB',
        tags=[Tags.AGENT, Tags.FUNCTION_CALLING, Tags.MULTI_MODAL, Tags.MULTI_TURN],
        description=_DESCRIPTION,
        dataset_id='https://github.com/ServiceNow/BrowserGym',
        subset_list=['default'],
        default_subset='default',
        eval_split='test',
        prompt_template='{question}',
        metric_list=['success_rate', 'error_rate'],
    )
)
class MiniWobAdapter(AgentLoopAdapter):
    """Run the pinned MiniWoB schedule through OpenEnv v0.4.1."""

    strategy_name = 'miniwob_openenv_function_calling'
    max_steps_default = MINIWOB_MAX_STEPS
    document_agent_config = False

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._validate_agent_config()
        self._environment_config = self._resolve_task_environment_config()
        self.observation_mode = self._environment_config.observation_mode or 'axtree_screenshot'
        if self.observation_mode not in {'axtree', 'axtree_screenshot'}:
            raise ValueError(f'Unsupported MiniWoB observation_mode: {self.observation_mode}')
        self._emit_profile_warning()

    def load(self) -> tuple[DatasetDict, None]:
        """Generate the deterministic episode schedule from pinned BrowserGym metadata."""
        records, metadata_path = load_miniwob_records()
        dataset = build_dataset_from_records(
            records=records,
            sample_fields=self.record_to_sample,
            name='test',
            location=str(metadata_path),
            limit=self.limit,
            repeats=1,
            shuffle=self.shuffle,
            seed=self.seed,
        )
        return DatasetDict({'default': dataset}), None

    def record_to_sample(self, record: Dict[str, Any]) -> Sample:
        """Convert one generated episode into an agent sample."""
        metadata = {
            **record,
            **self._report_metadata({'observation_mode': self.observation_mode}),
        }
        return Sample(
            input='The task goal and browser observation are supplied when the OpenEnv episode is reset.',
            target='1',
            tools=[BROWSER_ACTION_TOOL_INFO],
            metadata=metadata,
        )

    def run_inference(self, model: Model, sample: Sample, output_dir: str, **kwargs: Any) -> TaskState:
        """Prepare per-sample artifact state before starting the OpenEnv episode."""
        task_slug = sample.metadata['task_id'].replace('.', '-')
        output_path = Path(output_dir)
        run_root = output_path
        for candidate in (output_path, *output_path.parents):
            if candidate.name == 'predictions':
                run_root = candidate.parent
                break
        artifact_dir = run_root / 'artifacts' / 'miniwob' / f'{sample.id}-{task_slug}'
        artifact_dir.mkdir(parents=True, exist_ok=True)
        sample.metadata.update({
            'artifact_dir': str(artifact_dir.resolve()),
            'observation_mode': self.observation_mode,
            'reward': 0.0,
            'done': False,
            'success': False,
            'runtime_error': None,
        })
        if isinstance(sample.input, str):
            sample.input = [ChatMessageUser(content=sample.input)]
        return super().run_inference(model, sample, output_dir, **kwargs)

    def build_strategy(self, sample: Sample) -> _MiniWobStrategy:
        """Build the restricted MiniWoB function-calling strategy."""
        return _MiniWobStrategy(
            system_prompt=(
                'Complete the browser task using the single tool named browser_action. Never emit click, fill, press, '
                'or any BrowserGym action as a tool name; those functions belong only inside browser_action.action. '
                'Inspect the available screenshot and accessibility tree before every action, then call browser_action '
                'exactly once per turn. IMPORTANT: click accepts only a string BID from the accessibility tree. Use '
                'mouse_click for screenshot coordinates. Coordinates are absolute screenshot pixels, not normalized '
                '0-1000 values; convert normalized coordinates using the screenshot width and height shown in the '
                'observation. '
                'The environment, not a textual answer, determines success.'
            )
        )

    def _build_tools_for_session(
        self,
        sample: Sample,
        session: TaskEnvironmentSession,
    ) -> Dict[str, Any]:
        """Return the model-facing OpenEnv browser action handler."""

        async def run_browser_action(call: ToolCall, _: Any) -> ToolExecutionOutput:
            action = validate_browser_action(str(call.function.arguments.get('action', '')))
            try:
                result = await session.step({'action_str': action})
                observation = self._merge_step_result(sample, result, action=action)
                return self._tool_output_from_observation(sample, observation)
            except Exception as exc:
                error = f'OpenEnv step failed: {exc}'
                sample.metadata['runtime_error'] = error
                return ToolExecutionOutput(
                    text=error,
                    metadata={'runtime_error': error},
                    terminate=True,
                    final_answer='0',
                )

        return {'browser_action': run_browser_action}

    def build_task_environment(
        self,
        sample: Sample,
    ) -> tuple[EnvironmentRuntimeLease, TaskEnvironmentSession]:
        """Start the configured service runtime and create one OpenEnv session."""
        config = self._environment_config
        runtime_cls = get_environment_runtime(config.runtime.name)
        runtime = runtime_cls()
        image = self._prepare_runtime_image()
        env_vars = {
            'BROWSERGYM_BENCHMARK': 'miniwob',
            'BROWSERGYM_HEADLESS': 'true',
            'BROWSERGYM_INCLUDE_SCREENSHOT': str(self.observation_mode == 'axtree_screenshot').lower(),
            'MAX_CONCURRENT_ENVS': '1',
        }
        lease = AsyncioLoopRunner.run(
            runtime.start(
                image=image,
                env_vars=env_vars,
                config=dict(config.runtime.config),
            )
        )
        try:
            backend_cls = get_task_environment(config.backend)
            session = backend_cls().create_session(
                base_url=lease.base_url,
                config=dict(config.backend_args),
            )
        except Exception:
            AsyncioLoopRunner.run(lease.close())
            raise
        sample.metadata['runtime_mode'] = 'local'
        return lease, session

    def _on_inference(self, model: Model, sample: Sample) -> InferenceResult:
        """Reset OpenEnv, drive AgentLoop, and convert terminal reward into a prediction."""
        from evalscope.api.agent import run_agent_loop

        with _EPISODE_SEMAPHORE:
            lease, session = self.build_task_environment(sample)
            try:
                reset_started = time.perf_counter()
                try:
                    reset_result = AsyncioLoopRunner.run(
                        session.reset(
                            seed=sample.metadata['seed'],
                            task_name=sample.metadata['openenv_task_name'],
                        )
                    )
                    initial_observation = self._merge_step_result(sample, reset_result)
                    if self.observation_mode == 'axtree_screenshot' and not initial_observation.get('screenshot'):
                        raise RuntimeError('OpenEnv did not return a screenshot for axtree_screenshot mode.')
                    reset_message = self._observation_message(sample, initial_observation)
                    initial_messages = [reset_message]
                except Exception as exc:
                    return self._failure(
                        model,
                        sample,
                        f'OpenEnv reset failed: {exc}',
                        source='openenv_runtime',
                    )
                reset_latency_ms = (time.perf_counter() - reset_started) * 1000

                strategy = self.build_strategy(sample)
                try:
                    result: AgentLoopResult = run_agent_loop(
                        model=model,
                        strategy=strategy,
                        handlers=self._build_tools_for_session(sample, session),
                        environment=None,
                        initial_messages=initial_messages,
                        all_tools=list(sample.tools or []),
                        max_steps=MINIWOB_MAX_STEPS,
                        sample_id=sample.id,
                        trace_strategy_name=strategy.name,
                        trace_env_name=None,
                        close_environment=False,
                    )
                except Exception as exc:
                    failure = self._failure(model, sample, f'Model inference failed: {exc}', source='model')
                    reset_metadata = reset_message.metadata or {}
                    failure.trace.events.insert(
                        0,
                        AgentTraceEvent(
                            step=0,
                            type=EventType.ENV_RESET,
                            message_id=reset_message.id,
                            latency_ms=reset_latency_ms,
                            payload={
                                'reward': reset_metadata.get('reward'),
                                'done': reset_metadata.get('done'),
                                'error': reset_metadata.get('error'),
                                'screenshot_path': reset_metadata.get('screenshot_path'),
                            },
                        ),
                    )
                    failure.messages.insert(0, reset_message)
                    return failure

                initial_metadata = reset_message.metadata or {}
                result.trace.task_environment = session.backend_name
                result.trace.task_environment_runtime = lease.name
                result.trace.events.insert(
                    0,
                    AgentTraceEvent(
                        step=0,
                        type=EventType.ENV_RESET,
                        message_id=reset_message.id,
                        latency_ms=reset_latency_ms,
                        payload={
                            'reward': initial_metadata.get('reward'),
                            'done': initial_metadata.get('done'),
                            'error': initial_metadata.get('error'),
                            'screenshot_path': initial_metadata.get('screenshot_path'),
                        },
                    ),
                )
                if sample.metadata['runtime_error']:
                    result.trace.add_event(
                        step=int(sample.metadata.get('browser_step', 0)),
                        type=EventType.ERROR,
                        payload={
                            'source': 'openenv_runtime',
                            'message': sample.metadata['runtime_error'],
                        },
                    )
                output = result.final_output
                output.completion = '1' if sample.metadata['success'] else '0'
                if sample.metadata['runtime_error']:
                    output.error = sample.metadata['runtime_error']
                output.metadata = {
                    **(output.metadata or {}),
                    **self._report_metadata(sample),
                }
                return InferenceResult(output=output, messages=result.messages, trace=result.trace)
            finally:
                if not sample.metadata['success']:
                    log_path = Path(sample.metadata['artifact_dir']) / 'openenv-container.log'
                    try:
                        AsyncioLoopRunner.run(lease.capture_logs(log_path))
                    except Exception as exc:
                        logger.warning(f'Failed to capture MiniWoB container logs: {exc}')
                try:
                    AsyncioLoopRunner.run(session.close())
                except Exception as exc:
                    logger.warning(f'Failed to close MiniWoB OpenEnv session: {exc}')
                try:
                    AsyncioLoopRunner.run(lease.close())
                except Exception as exc:
                    logger.warning(f'Failed to close MiniWoB environment runtime: {exc}')

    def match_score(
        self,
        original_prediction: str,
        filtered_prediction: str,
        reference: str,
        task_state: TaskState,
    ) -> Score:
        """Score official environment success separately from runtime reliability."""
        runtime_error = task_state.metadata.get('runtime_error')
        return Score(
            extracted_prediction=filtered_prediction,
            prediction=original_prediction,
            value={
                'success_rate': float(bool(task_state.metadata.get('success'))),
                'error_rate': float(bool(runtime_error)),
            },
            main_score_name='success_rate',
            metadata={
                **self._report_metadata(task_state.metadata),
                'runtime_error': runtime_error,
            },
        )

    def _on_generate_report(self, scores: Dict[str, List[AggScore]], model_name: str) -> Report:
        report = super()._on_generate_report(scores, model_name)
        return _MiniWobReport.model_validate({
            **report.model_dump(exclude={'num'}),
            'metadata': self._report_metadata({
                'runtime_mode': 'local',
                'observation_mode': self.observation_mode,
            }),
        })

    def _validate_agent_config(self) -> None:
        agent_config = self._task_config.agent_config if self._task_config is not None else None
        if agent_config is None:
            return
        if not isinstance(agent_config, NativeAgentConfig):
            raise ValueError('MiniWoB supports only EvalScope native AgentLoop agents.')
        if agent_config.environment is not None or agent_config.environment_extra:
            raise ValueError(
                'MiniWoB does not use an Agent execution environment; configure agent_config.task_environment instead.'
            )
        if 'max_steps' in agent_config.model_fields_set and agent_config.max_steps != MINIWOB_MAX_STEPS:
            raise ValueError(f'MiniWoB fixes max_steps={MINIWOB_MAX_STEPS}.')
        if 'strategy' in agent_config.model_fields_set:
            raise ValueError('MiniWoB fixes its internal strategy; omit agent_config.strategy.')
        if agent_config.tools or agent_config.mcp_servers:
            raise ValueError('MiniWoB exposes only its benchmark-owned browser_action tool.')

    def _resolve_task_environment_config(self) -> TaskEnvironmentConfig:
        agent_config = self._task_config.agent_config if self._task_config is not None else None
        config = agent_config.task_environment if isinstance(agent_config, NativeAgentConfig) else None
        if config is None:
            return TaskEnvironmentConfig.model_validate({
                'backend': 'openenv',
                'observation_mode': 'axtree_screenshot',
                'runtime': {
                    'name': 'ms_enclave_docker',
                    'config': {},
                },
            })
        if config.backend != 'openenv':
            raise ValueError("MiniWoB task_environment.backend must be 'openenv'.")
        if config.runtime.name != 'ms_enclave_docker':
            raise ValueError("MiniWoB supports only task_environment.runtime.name='ms_enclave_docker'.")
        return config

    def _prepare_runtime_image(self) -> str:
        global _RUNTIME_IMAGE_TAG
        with _IMAGE_BUILD_LOCK:
            if _RUNTIME_IMAGE_TAG is not None:
                return _RUNTIME_IMAGE_TAG
            runtime_dir = Path(__file__).parent / 'runtime'
            result = prepare_docker_image(
                DockerImageSpec(
                    name_prefix='evalscope-openenv-browsergym',
                    context_dir=str(runtime_dir),
                    dockerfile='Dockerfile',
                    build_args={
                        'OPENENV_COMMIT': OPENENV_COMMIT,
                        'MINIWOB_COMMIT': MINIWOB_COMMIT,
                        'EVALSCOPE_PIP_INDEX_URL': os.environ.get(
                            'EVALSCOPE_PIP_INDEX_URL',
                            RUNTIME_PIP_INDEX_URL,
                        ),
                    },
                )
            )
            _RUNTIME_IMAGE_TAG = result.image_tag
            return result.image_tag

    def _merge_step_result(
        self,
        sample: Sample,
        result: Any,
        *,
        action: Optional[str] = None,
    ) -> Dict[str, Any]:
        observation = dict(result.observation)
        openenv_metadata = result.metadata if isinstance(result.metadata, dict) else {}
        browsergym_obs = openenv_metadata.get('browsergym_obs')
        raw_action_error = browsergym_obs.get('last_action_error') if isinstance(browsergym_obs, dict) else None
        if observation.get('last_action_error') and not observation.get('error') and isinstance(raw_action_error, str):
            observation['error'] = raw_action_error
        step = int(sample.metadata.get('browser_step', 0)) + (1 if action is not None else 0)
        reward = float(result.reward or 0.0)
        done = bool(result.done)
        success = done and reward > 0
        sample.metadata.update({
            'reward': reward,
            'done': done,
            'success': success,
            'browser_step': step,
            'last_action_error': bool(observation.get('last_action_error')),
        })
        observation.update({
            'reward': reward,
            'done': done,
            'step': step,
        })
        return observation

    def _tool_output_from_observation(self, sample: Sample, observation: Dict[str, Any]) -> ToolExecutionOutput:
        screenshot = self._save_screenshot(sample, observation)
        attachments = [ContentImage(image=str(screenshot))] if screenshot is not None else []
        return ToolExecutionOutput(
            text=self._format_observation(observation),
            attachments=attachments,
            metadata=self._observation_metadata(sample, observation, screenshot),
            terminate=bool(observation['done']),
            final_answer='1' if sample.metadata['success'] else '0',
        )

    def _observation_message(self, sample: Sample, observation: Dict[str, Any]) -> ChatMessageUser:
        text = self._format_observation(observation)
        screenshot = self._save_screenshot(sample, observation)
        metadata = self._observation_metadata(sample, observation, screenshot)
        if screenshot is None:
            return ChatMessageUser(content=text, metadata=metadata)
        return ChatMessageUser(
            content=[ContentText(text=text), ContentImage(image=str(screenshot))],
            metadata=metadata,
        )

    @staticmethod
    def _format_observation(observation: Dict[str, Any]) -> str:
        error = str(observation.get('error') or '')
        fields = [
            f"Goal: {observation.get('goal', '')}",
            f"URL: {observation.get('url', '')}",
            f"Step: {observation.get('step', 0)}",
            f"Reward: {observation.get('reward', 0.0)}",
            f"Done: {bool(observation.get('done'))}",
            f"Last action error: {bool(observation.get('last_action_error'))}",
        ]
        screenshot = observation.get('screenshot')
        if screenshot is not None:
            try:
                height = len(screenshot)
                width = len(screenshot[0]) if height else 0
            except (TypeError, IndexError):
                height = width = 0
            if width and height:
                fields.append(
                    f'Screenshot size: {width}x{height} pixels. Coordinate actions require absolute screenshot pixels, '
                    'not normalized 0-1000 coordinates.'
                )
        if error:
            fields.append(f'Environment message: {error}')
        fields.extend(['Accessibility tree:', str(observation.get('axtree_txt') or observation.get('text') or '')])
        return '\n'.join(fields)

    def _save_screenshot(self, sample: Sample, observation: Dict[str, Any]) -> Optional[Path]:
        if self.observation_mode != 'axtree_screenshot':
            return None
        pixels = observation.get('screenshot')
        if pixels is None:
            raise RuntimeError('OpenEnv screenshot is missing.')
        array = np.asarray(pixels, dtype=np.uint8)
        if array.ndim != 3 or array.shape[2] not in {3, 4}:
            raise ValueError(f'Unexpected OpenEnv screenshot shape: {array.shape}.')
        path = Path(sample.metadata['artifact_dir']) / f"step-{int(observation.get('step', 0)):03d}.png"
        Image.fromarray(array).save(path, format='PNG')
        return path

    @staticmethod
    def _observation_metadata(
        sample: Sample,
        observation: Dict[str, Any],
        screenshot: Optional[Path],
    ) -> Dict[str, Any]:
        return {
            'reward': float(observation['reward']),
            'done': bool(observation['done']),
            'success': bool(sample.metadata['success']),
            'step': int(observation['step']),
            'last_action_error': bool(observation.get('last_action_error')),
            'error': str(observation.get('error') or ''),
            'screenshot_path': str(screenshot) if screenshot is not None else None,
            'profile': MINIWOB_PROFILE,
        }

    def _failure(self, model: Model, sample: Sample, error: str, *, source: str) -> InferenceResult:
        error_key = 'runtime_error' if source == 'openenv_runtime' else 'model_error'
        sample.metadata.update({
            error_key: error,
            'reward': 0.0,
            'done': False,
            'success': False,
        })
        output = ModelOutput.from_content(model=model.name, content='0')
        output.error = error
        output.metadata = {
            **self._report_metadata(sample),
            error_key: error,
        }
        message = ChatMessageUser(content=error)
        trace = AgentTrace(
            strategy='miniwob_openenv_function_calling',
            task_environment='openenv',
            task_environment_runtime=self._environment_config.runtime.name,
            max_steps=MINIWOB_MAX_STEPS,
        )
        trace.add_event(
            step=int(sample.metadata.get('browser_step', 0)),
            type=EventType.ERROR,
            message_id=message.id,
            payload={
                'source': source,
                'message': error,
            },
        )
        return InferenceResult(output=output, messages=[message], trace=trace)

    @staticmethod
    def _report_metadata(sample_or_metadata: Sample | Dict[str, Any]) -> Dict[str, Any]:
        metadata = sample_or_metadata.metadata if isinstance(sample_or_metadata, Sample) else sample_or_metadata
        return {
            'profile': MINIWOB_PROFILE,
            'max_steps': MINIWOB_MAX_STEPS,
            'official_browsergym_action_config': True,
            'official_browsergym_evaluation_protocol': False,
            'openenv_version': OPENENV_VERSION,
            'openenv_commit': OPENENV_COMMIT,
            'openenv_patch_sha256': OPENENV_PATCH_SHA256,
            'browsergym_version': BROWSERGYM_VERSION,
            'browsergym_commit': BROWSERGYM_COMMIT,
            'miniwob_commit': MINIWOB_COMMIT,
            'csv_sha256': BROWSERGYM_METADATA_SHA256,
            'runtime_mode': metadata.get('runtime_mode'),
            'observation_mode': metadata.get('observation_mode'),
        }

    @staticmethod
    def _emit_profile_warning() -> None:
        global _PROFILE_WARNING_EMITTED
        with _PROFILE_WARNING_LOCK:
            if _PROFILE_WARNING_EMITTED:
                return
            logger.warning(
                'MiniWoB uses BrowserGym 0.14.3 miniwob_all actions through a pinned EvalScope patch to OpenEnv v0.4.1, '
                'but its 20-step budget differs from BrowserGym Experiments. Scores are not directly comparable with '
                'the official leaderboard protocol.'
            )
            _PROFILE_WARNING_EMITTED = True


__all__ = ['MiniWobAdapter']
