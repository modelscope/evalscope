"""Reusable adapter flow for benchmarks served through OpenEnv."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from evalscope.api.agent import AgentTrace, AgentTraceEvent, EventType, NativeAgentConfig
from evalscope.api.environment import (
    EnvironmentRuntimeHandle,
    EnvironmentStepResult,
    TaskEnvironmentConfig,
    TaskEnvironmentSession,
)
from evalscope.api.evaluator import InferenceResult, TaskState
from evalscope.api.messages import ChatMessage, ChatMessageUser
from evalscope.api.metric import Score
from evalscope.api.model import Model, ModelOutput
from evalscope.api.registry import get_environment_runtime, get_task_environment
from evalscope.utils.asyncio_runtime import AsyncioLoopRunner
from evalscope.utils.logger import get_logger
from .agent_adapter import AgentLoopAdapter

logger = get_logger()


class OpenEnvAdapter(AgentLoopAdapter):
    """Run an AgentLoop against one stateful OpenEnv episode per sample.

    Subclasses provide benchmark data plus five environment-specific hooks:
    runtime image, runtime environment variables, reset arguments, model-facing
    tools and observation formatting. This class owns the common service/session
    lifecycle, AgentLoop invocation, trace wiring, error classification and
    reward-based scoring.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._validate_agent_config()
        self.task_environment_config = self._resolve_task_environment_config()
        self.validate_task_environment_config(self.task_environment_config)

    def default_task_environment_config(self) -> TaskEnvironmentConfig:
        """Return the benchmark's default OpenEnv backend and runtime."""
        raise NotImplementedError

    def validate_task_environment_config(self, config: TaskEnvironmentConfig) -> None:
        """Validate benchmark-specific task-environment constraints."""
        if config.backend != 'openenv':
            raise ValueError("task_environment.backend must be 'openenv'.")

    def prepare_task_environment_image(self) -> Optional[str]:
        """Build or resolve the service image passed to the runtime."""
        return None

    def task_environment_env_vars(self, sample: Any) -> Dict[str, str]:
        """Return environment variables for the per-sample service."""
        return {}

    def task_reset_kwargs(self, sample: Any) -> Dict[str, Any]:
        """Return keyword arguments forwarded to ``session.reset``."""
        return {}

    def build_task_tools(self, sample: Any, session: TaskEnvironmentSession) -> Dict[str, Any]:
        """Return model-facing tool handlers bound to the active session."""
        raise NotImplementedError

    def task_observation_messages(
        self,
        sample: Any,
        observation: Dict[str, Any],
    ) -> List[ChatMessage]:
        """Convert the reset observation into initial AgentLoop messages."""
        raise NotImplementedError

    def normalize_task_observation(
        self,
        result: EnvironmentStepResult,
    ) -> Dict[str, Any]:
        """Normalize one backend result before exposing it to the benchmark."""
        return dict(result.observation)

    def is_task_success(self, result: EnvironmentStepResult) -> bool:
        """Return whether an environment result represents task success."""
        return bool(result.done and (result.reward or 0) > 0)

    def process_task_result(
        self,
        sample: Any,
        result: EnvironmentStepResult,
        *,
        action_performed: bool,
    ) -> Dict[str, Any]:
        """Update common episode state and return an enriched observation."""
        observation = self.normalize_task_observation(result)
        step = int(sample.metadata.get('task_step', 0)) + int(action_performed)
        reward = float(result.reward or 0.0)
        done = bool(result.done)
        success = self.is_task_success(result)
        sample.metadata.update({
            'reward': reward,
            'done': done,
            'success': success,
            'task_step': step,
        })
        observation.update({
            'reward': reward,
            'done': done,
            'step': step,
        })
        return observation

    def run_inference(self, model: Model, sample: Any, output_dir: str, **kwargs: Any) -> TaskState:
        """Initialize per-episode state and artifact storage."""
        artifact_dir = self._artifact_dir(sample, output_dir)
        artifact_dir.mkdir(parents=True, exist_ok=True)
        sample.metadata.update({
            'artifact_dir': str(artifact_dir.resolve()),
            'reward': 0.0,
            'done': False,
            'success': False,
            'runtime_error': None,
            'model_error': None,
            'task_step': 0,
        })
        return super().run_inference(model, sample, output_dir, **kwargs)

    def _on_inference(self, model: Model, sample: Any) -> InferenceResult:
        handle: Optional[EnvironmentRuntimeHandle] = None
        session: Optional[TaskEnvironmentSession] = None
        try:
            try:
                handle, session = self.start_task_environment(sample)
                reset_started = time.perf_counter()
                reset_result = AsyncioLoopRunner.run(session.reset(**self.task_reset_kwargs(sample)))
                reset_latency_ms = (time.perf_counter() - reset_started) * 1000
                observation = self.process_task_result(sample, reset_result, action_performed=False)
                initial_messages = self.task_observation_messages(sample, observation)
                if not initial_messages:
                    raise ValueError('task_observation_messages must return at least one message.')
            except Exception as exc:
                return self._task_failure(model, sample, f'OpenEnv reset failed: {exc}', source='openenv_runtime')

            agent_config = self._task_config.agent_config if self._task_config is not None else None
            strategy = self._resolve_strategy(sample, agent_config)
            max_steps = self._resolve_max_steps(agent_config)
            try:
                from evalscope.api.agent import run_agent_loop

                result = run_agent_loop(
                    model=model,
                    strategy=strategy,
                    handlers=self.build_task_tools(sample, session),
                    environment=None,
                    initial_messages=initial_messages,
                    all_tools=list(sample.tools or []),
                    max_steps=max_steps,
                    sample_id=sample.id,
                    trace_strategy_name=getattr(strategy, 'name', None),
                    trace_env_name=None,
                    mcp_configs=self._resolve_mcp_configs(agent_config),
                )
            except Exception as exc:
                failure = self._task_failure(model, sample, f'Model inference failed: {exc}', source='model')
                self._attach_task_trace(failure.trace, initial_messages[0], reset_latency_ms, handle, session)
                failure.messages = initial_messages + failure.messages
                return failure

            self._attach_task_trace(result.trace, initial_messages[0], reset_latency_ms, handle, session)
            if sample.metadata.get('runtime_error'):
                result.trace.add_event(
                    step=int(sample.metadata.get('task_step', 0)),
                    type=EventType.ERROR,
                    payload={
                        'source': 'openenv_runtime',
                        'message': sample.metadata['runtime_error'],
                    },
                )
            result.final_output.completion = '1' if sample.metadata['success'] else '0'
            if sample.metadata.get('runtime_error'):
                result.final_output.error = sample.metadata['runtime_error']
            result.final_output.metadata = {
                **(result.final_output.metadata or {}),
                **self._task_output_metadata(sample),
            }
            return InferenceResult(output=result.final_output, messages=result.messages, trace=result.trace)
        finally:
            if handle is not None:
                self._close_task_environment(sample, handle, session)

    def start_task_environment(
        self,
        sample: Any,
    ) -> tuple[EnvironmentRuntimeHandle, TaskEnvironmentSession]:
        """Start the configured runtime and bind an OpenEnv session to it."""
        config = self.task_environment_config
        runtime = get_environment_runtime(config.runtime.name)()
        handle = AsyncioLoopRunner.run(
            runtime.start(
                image=self.prepare_task_environment_image(),
                env_vars=self.task_environment_env_vars(sample),
                config=dict(config.runtime.config),
            )
        )
        try:
            backend = get_task_environment(config.backend)()
            session = backend.create_session(base_url=handle.base_url, config=dict(config.backend_args))
        except Exception:
            AsyncioLoopRunner.run(handle.close())
            raise
        sample.metadata['task_environment'] = config.backend
        sample.metadata['task_environment_runtime'] = handle.name
        return handle, session

    def match_score(
        self,
        original_prediction: str,
        filtered_prediction: str,
        reference: str,
        task_state: TaskState,
    ) -> Score:
        """Score environment success separately from runtime reliability."""
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
                **self._task_output_metadata(task_state.metadata),
                'runtime_error': runtime_error,
            },
        )

    def _validate_agent_config(self) -> None:
        agent_config = self._task_config.agent_config if self._task_config is not None else None
        if agent_config is None:
            return
        if not isinstance(agent_config, NativeAgentConfig):
            raise ValueError('OpenEnv benchmarks support only EvalScope native AgentLoop agents.')
        if agent_config.environment is not None or agent_config.environment_extra:
            raise ValueError(
                'OpenEnv benchmarks do not use an Agent execution environment; '
                'configure agent_config.task_environment instead.'
            )

    def _resolve_task_environment_config(self) -> TaskEnvironmentConfig:
        agent_config = self._task_config.agent_config if self._task_config is not None else None
        if isinstance(agent_config, NativeAgentConfig) and agent_config.task_environment is not None:
            return agent_config.task_environment
        return self.default_task_environment_config()

    def _attach_task_trace(
        self,
        trace: AgentTrace,
        reset_message: ChatMessage,
        reset_latency_ms: float,
        handle: EnvironmentRuntimeHandle,
        session: TaskEnvironmentSession,
    ) -> None:
        trace.task_environment = session.backend_name
        trace.task_environment_runtime = handle.name
        metadata = reset_message.metadata or {}
        trace.events.insert(
            0,
            AgentTraceEvent(
                step=0,
                type=EventType.ENV_RESET,
                message_id=reset_message.id,
                latency_ms=reset_latency_ms,
                payload={
                    'reward': metadata.get('reward'),
                    'done': metadata.get('done'),
                    'error': metadata.get('error'),
                    'screenshot_path': metadata.get('screenshot_path'),
                },
            ),
        )

    def _task_failure(self, model: Model, sample: Any, error: str, *, source: str) -> InferenceResult:
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
            **self._task_output_metadata(sample),
            error_key: error,
        }
        message = ChatMessageUser(content=error)
        trace = AgentTrace(
            strategy=self.strategy_name,
            task_environment=self.task_environment_config.backend,
            task_environment_runtime=self.task_environment_config.runtime.name,
            max_steps=self._resolve_max_steps(self._task_config.agent_config if self._task_config else None),
        )
        trace.add_event(
            step=int(sample.metadata.get('task_step', 0)),
            type=EventType.ERROR,
            message_id=message.id,
            payload={
                'source': source,
                'message': error,
            },
        )
        return InferenceResult(output=output, messages=[message], trace=trace)

    @staticmethod
    def _task_output_metadata(sample_or_metadata: Any) -> Dict[str, Any]:
        metadata = sample_or_metadata.metadata if hasattr(sample_or_metadata, 'metadata') else sample_or_metadata
        return {
            'reward': float(metadata.get('reward', 0.0)),
            'done': bool(metadata.get('done')),
            'success': bool(metadata.get('success')),
            'task_environment': metadata.get('task_environment'),
            'task_environment_runtime': metadata.get('task_environment_runtime'),
        }

    def _close_task_environment(
        self,
        sample: Any,
        handle: EnvironmentRuntimeHandle,
        session: Optional[TaskEnvironmentSession],
    ) -> None:
        if not sample.metadata.get('success'):
            log_path = Path(sample.metadata['artifact_dir']) / 'task-environment.log'
            try:
                AsyncioLoopRunner.run(handle.capture_logs(log_path))
            except Exception as exc:
                logger.warning(f'Failed to capture task-environment logs: {exc}')
        if session is not None:
            try:
                AsyncioLoopRunner.run(session.close())
            except Exception as exc:
                logger.warning(f'Failed to close task-environment session: {exc}')
        try:
            AsyncioLoopRunner.run(handle.close())
        except Exception as exc:
            logger.warning(f'Failed to close task-environment runtime: {exc}')

    def _artifact_dir(self, sample: Any, output_dir: str) -> Path:
        output_path = Path(output_dir)
        run_root = output_path
        for candidate in (output_path, *output_path.parents):
            if candidate.name == 'predictions':
                run_root = candidate.parent
                break
        task_name = str(sample.metadata.get('task_id') or sample.id or 'task').replace('.', '-')
        return run_root / 'artifacts' / self.name / f'{sample.id}-{task_name}'


__all__ = ['OpenEnvAdapter']
