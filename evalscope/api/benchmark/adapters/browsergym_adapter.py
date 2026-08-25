"""Reusable direct BrowserGym benchmark flow."""

from __future__ import annotations

import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
from PIL import Image

from evalscope.agent.strategies.function_calling import FunctionCallingStrategy
from evalscope.api.agent import AgentTrace, AgentTraceEvent, EventType, NativeAgentConfig, ToolExecutionOutput
from evalscope.api.dataset import Sample
from evalscope.api.evaluator import InferenceResult, TaskState
from evalscope.api.messages import ChatMessage, ChatMessageUser, ContentImage, ContentText
from evalscope.api.metric import Score
from evalscope.api.model import Model, ModelOutput
from evalscope.api.tool import ToolCall
from evalscope.utils.asyncio_runtime import AsyncioLoopRunner
from evalscope.utils.import_utils import check_import
from evalscope.utils.logger import get_logger

from .agent_adapter import AgentLoopAdapter

# BrowserGym owns Playwright objects that must be created, used and closed on
# the same thread. A single shared worker preserves that thread affinity.
_BROWSERGYM_EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix='evalscope-browsergym')
logger = get_logger()


@dataclass
class BrowserGymStep:
    """Normalized result of one BrowserGym reset or step."""

    observation: Dict[str, Any]
    reward: float = 0.0
    done: bool = False


class BrowserGymSession:
    """One BrowserGym episode executed on the shared Playwright thread."""

    def __init__(
        self,
        *,
        task_name: str,
        seed: int,
        action_mapping: Any,
        max_steps: int,
        task_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.task_name = task_name
        self.seed = seed
        self.action_mapping = action_mapping
        self.max_steps = max_steps
        self.task_kwargs = dict(task_kwargs or {})
        self._env: Any = None
        self._closed = False

    async def reset(self) -> BrowserGymStep:
        """Create and reset the BrowserGym environment."""
        return await self._run(self._reset)

    async def step(self, action: str) -> BrowserGymStep:
        """Execute one BrowserGym action."""
        return await self._run(self._step, action)

    async def close(self) -> None:
        """Close this BrowserGym episode."""
        if self._closed:
            return
        self._closed = True
        await self._run(self._close, allow_closed=True)

    async def _run(self, func: Any, *args: Any, allow_closed: bool = False) -> Any:
        if self._closed and not allow_closed:
            raise RuntimeError('BrowserGym session is already closed.')
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(_BROWSERGYM_EXECUTOR, func, *args)

    def _reset(self) -> BrowserGymStep:
        import gymnasium as gym

        self._env = gym.make(
            f'browsergym/{self.task_name}',
            disable_env_checker=True,
            max_episode_steps=self.max_steps,
            headless=True,
            action_mapping=self.action_mapping,
            task_kwargs=self.task_kwargs,
        )
        observation, _ = self._env.reset(seed=self.seed)
        return self._normalize(observation)

    def _step(self, action: str) -> BrowserGymStep:
        if self._env is None:
            raise RuntimeError('BrowserGym session must be reset before step.')
        observation, reward, terminated, truncated, _ = self._env.step(action)
        return self._normalize(
            observation,
            reward=reward,
            done=bool(terminated or truncated),
        )

    def _close(self) -> None:
        if self._env is not None:
            self._env.close()
            self._env = None

    @staticmethod
    def _normalize(
        observation: Any,
        *,
        reward: Any = 0.0,
        done: bool = False,
    ) -> BrowserGymStep:
        if not isinstance(observation, dict):
            raise TypeError(f'BrowserGym observation must be a dictionary, got {type(observation).__name__}.')
        normalized = dict(observation)
        axtree_object = normalized.pop('axtree_object', None)
        normalized.pop('dom_object', None)
        if 'axtree_txt' not in normalized and axtree_object is not None:
            from browsergym.utils.obs import flatten_axtree_to_str

            normalized['axtree_txt'] = flatten_axtree_to_str(axtree_object)
        return BrowserGymStep(
            observation=normalized,
            reward=float(reward or 0.0),
            done=done,
        )


class BrowserGymAdapter(AgentLoopAdapter):
    """Run EvalScope AgentLoop directly against BrowserGym.

    Subclasses provide the BrowserGym module and action subset, plus normal
    benchmark dataset hooks. This class owns the BrowserGym lifecycle,
    observation formatting, trace wiring and reward-based scoring.
    """

    browsergym_module: str = ''
    browsergym_action_subset: str = ''
    browsergym_system_prompt: Optional[str] = None

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        if not self.browsergym_module or not self.browsergym_action_subset:
            raise ValueError('BrowserGym adapters must define module and action subset.')
        check_import(
            self.browsergym_module,
            extra=self.name,
            raise_error=True,
            feature_name=self.pretty_name,
        )
        self._validate_agent_config()

    def build_strategy(self, sample: Sample) -> FunctionCallingStrategy:
        """Build the standard single-action BrowserGym strategy."""
        return FunctionCallingStrategy(
            system_prompt=self.browsergym_system_prompt,
            include_submit_tool=False,
            max_tool_calls_per_turn=1,
        )

    def browsergym_task_kwargs(self, sample: Sample) -> Dict[str, Any]:
        """Return benchmark task constructor arguments."""
        return {}

    @staticmethod
    def validate_browser_action(action: str) -> str:
        """Validate and normalize one BrowserGym action expression."""
        action = action.strip()
        if not action:
            raise ValueError('browser_action must not be empty.')
        return action

    def create_browsergym_session(self, sample: Sample) -> BrowserGymSession:
        """Create one direct BrowserGym session for a sample."""
        from browsergym.core.action.highlevel import HighLevelActionSet

        action_mapping = HighLevelActionSet(
            subsets=[self.browsergym_action_subset],
            multiaction=False,
            strict=False,
            retry_with_force=True,
            demo_mode='off',
        ).to_python_code
        agent_config = self._task_config.agent_config if self._task_config is not None else None
        return BrowserGymSession(
            task_name=sample.metadata['task_name'],
            seed=int(sample.metadata['seed']),
            action_mapping=action_mapping,
            max_steps=self._resolve_max_steps(agent_config),
            task_kwargs=self.browsergym_task_kwargs(sample),
        )

    def build_browsergym_tools(
        self,
        sample: Sample,
        session: BrowserGymSession,
    ) -> Dict[str, Any]:
        """Bind the browser action tool to an active BrowserGym session."""

        async def handle_browser_action(call: ToolCall, _: Any) -> ToolExecutionOutput:
            try:
                action = self.validate_browser_action(str(call.function.arguments.get('action', '')))
                result = await session.step(action)
                observation = self._process_browsergym_result(sample, result, action_performed=True)
                return self._tool_output_from_observation(sample, observation)
            except Exception as exc:
                error = f'BrowserGym step failed: {exc}'
                sample.metadata['runtime_error'] = error
                return ToolExecutionOutput(
                    text=error,
                    metadata={'runtime_error': error},
                    terminate=True,
                    final_answer='0',
                )

        return {'browser_action': handle_browser_action}

    def run_inference(self, model: Model, sample: Sample, output_dir: str, **kwargs: Any) -> TaskState:
        """Initialize direct BrowserGym episode state and artifacts."""
        artifact_dir = self._artifact_dir(sample, output_dir)
        artifact_dir.mkdir(parents=True, exist_ok=True)
        sample.metadata.update({
            'artifact_dir': str(artifact_dir.resolve()),
            'reward': 0.0,
            'done': False,
            'success': False,
            'runtime_error': None,
            'model_error': None,
            'browser_step': 0,
        })
        return super().run_inference(model, sample, output_dir, **kwargs)

    def _on_inference(self, model: Model, sample: Sample) -> InferenceResult:
        session: Optional[BrowserGymSession] = None
        try:
            try:
                session = self.create_browsergym_session(sample)
                reset_started = time.perf_counter()
                reset_result = AsyncioLoopRunner.run(session.reset())
                reset_latency_ms = (time.perf_counter() - reset_started) * 1000
                observation = self._process_browsergym_result(sample, reset_result, action_performed=False)
                initial_message = self._observation_message(sample, observation)
            except Exception as exc:
                return self._browsergym_failure(
                    model,
                    sample,
                    f'BrowserGym reset failed: {exc}',
                    source='browsergym',
                )

            agent_config = self._task_config.agent_config if self._task_config is not None else None
            strategy = self._resolve_strategy(sample, agent_config)
            max_steps = self._resolve_max_steps(agent_config)
            try:
                from evalscope.api.agent import run_agent_loop

                result = run_agent_loop(
                    model=model,
                    strategy=strategy,
                    handlers=self.build_browsergym_tools(sample, session),
                    environment=None,
                    initial_messages=[initial_message],
                    all_tools=list(sample.tools or []),
                    max_steps=max_steps,
                    sample_id=sample.id,
                    trace_strategy_name=getattr(strategy, 'name', None),
                    trace_env_name='browsergym',
                )
            except Exception as exc:
                failure = self._browsergym_failure(
                    model,
                    sample,
                    f'Model inference failed: {exc}',
                    source='model',
                )
                self._attach_reset_trace(failure.trace, initial_message, reset_latency_ms)
                failure.messages.insert(0, initial_message)
                return failure

            self._attach_reset_trace(result.trace, initial_message, reset_latency_ms)
            if sample.metadata.get('runtime_error'):
                result.trace.add_event(
                    step=int(sample.metadata.get('browser_step', 0)),
                    type=EventType.ERROR,
                    payload={
                        'source': 'browsergym',
                        'message': sample.metadata['runtime_error'],
                    },
                )
            result.final_output.completion = '1' if sample.metadata['success'] else '0'
            if sample.metadata.get('runtime_error'):
                result.final_output.error = sample.metadata['runtime_error']
            result.final_output.metadata = {
                **(result.final_output.metadata or {}),
                **self._browsergym_output_metadata(sample),
            }
            return InferenceResult(output=result.final_output, messages=result.messages, trace=result.trace)
        finally:
            if session is not None:
                try:
                    AsyncioLoopRunner.run(session.close())
                except Exception as exc:
                    logger.warning(f'Failed to close BrowserGym session: {exc}')

    def match_score(
        self,
        original_prediction: str,
        filtered_prediction: str,
        reference: str,
        task_state: TaskState,
    ) -> Score:
        """Score BrowserGym success separately from runtime reliability."""
        runtime_error = task_state.metadata.get('runtime_error')
        return Score(
            extracted_prediction=filtered_prediction,
            prediction=original_prediction,
            value={
                'success_rate': float(bool(task_state.metadata.get('success'))),
                'error_rate': float(bool(runtime_error)),
            },
            main_score_name='success_rate',
            metadata={'runtime_error': runtime_error},
        )

    def _validate_agent_config(self) -> None:
        agent_config = self._task_config.agent_config if self._task_config is not None else None
        if agent_config is None:
            return
        if not isinstance(agent_config, NativeAgentConfig):
            raise ValueError('BrowserGym benchmarks support only EvalScope native AgentLoop agents.')
        if agent_config.environment is not None or agent_config.environment_extra:
            raise ValueError('BrowserGym benchmarks do not use an Agent command execution environment.')

    @staticmethod
    def _process_browsergym_result(
        sample: Sample,
        result: BrowserGymStep,
        *,
        action_performed: bool,
    ) -> Dict[str, Any]:
        observation = dict(result.observation)
        action_error = str(observation.get('last_action_error') or '')
        if action_error and not observation.get('error'):
            observation['error'] = action_error
        step = int(sample.metadata.get('browser_step', 0)) + int(action_performed)
        success = bool(result.done and result.reward > 0)
        sample.metadata.update({
            'reward': result.reward,
            'done': result.done,
            'success': success,
            'browser_step': step,
        })
        observation.update({
            'reward': result.reward,
            'done': result.done,
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
            array = np.asarray(screenshot)
            if array.ndim >= 2:
                height, width = array.shape[:2]
                fields.append(f'Screenshot size: {width}x{height} pixels.')
        error = str(observation.get('error') or '')
        if error:
            fields.append(f'Environment message: {error}')
        fields.extend(['Accessibility tree:', str(observation.get('axtree_txt') or observation.get('text') or '')])
        return '\n'.join(fields)

    @staticmethod
    def _save_screenshot(sample: Sample, observation: Dict[str, Any]) -> Optional[Path]:
        pixels = observation.get('screenshot')
        if pixels is None:
            return None
        array = np.asarray(pixels, dtype=np.uint8)
        if array.ndim != 3 or array.shape[2] not in {3, 4}:
            raise ValueError(f'Unexpected BrowserGym screenshot shape: {array.shape}.')
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
        }

    def _browsergym_failure(self, model: Model, sample: Sample, error: str, *, source: str) -> InferenceResult:
        error_key = 'runtime_error' if source == 'browsergym' else 'model_error'
        sample.metadata.update({
            error_key: error,
            'reward': 0.0,
            'done': False,
            'success': False,
        })
        output = ModelOutput.from_content(model=model.name, content='0')
        output.error = error
        output.metadata = {
            **self._browsergym_output_metadata(sample),
            error_key: error,
        }
        message = ChatMessageUser(content=error)
        trace = AgentTrace(
            strategy=self.strategy_name,
            environment='browsergym',
            max_steps=self._resolve_max_steps(self._task_config.agent_config if self._task_config else None),
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
    def _browsergym_output_metadata(sample: Sample) -> Dict[str, Any]:
        return {
            'reward': float(sample.metadata.get('reward', 0.0)),
            'done': bool(sample.metadata.get('done')),
            'success': bool(sample.metadata.get('success')),
            'browsergym_task_name': sample.metadata.get('task_name'),
        }

    @staticmethod
    def _attach_reset_trace(
        trace: AgentTrace,
        reset_message: ChatMessage,
        reset_latency_ms: float,
    ) -> None:
        metadata = reset_message.metadata or {}
        trace.events.insert(
            0,
            AgentTraceEvent(
                step=0,
                type=EventType.ENV_RESET,
                message_id=reset_message.id,
                latency_ms=reset_latency_ms,
                payload={
                    'backend': 'browsergym',
                    'reward': metadata.get('reward'),
                    'done': metadata.get('done'),
                    'error': metadata.get('error'),
                    'screenshot_path': metadata.get('screenshot_path'),
                },
            ),
        )

    def _artifact_dir(self, sample: Sample, output_dir: str) -> Path:
        output_path = Path(output_dir)
        run_root = output_path
        for candidate in (output_path, *output_path.parents):
            if candidate.name == 'predictions':
                run_root = candidate.parent
                break
        task_name = str(sample.metadata.get('task_id') or sample.id or 'task').replace('.', '-')
        return run_root / 'artifacts' / self.name / f'{sample.id}-{task_name}'


__all__ = ['BrowserGymAdapter', 'BrowserGymSession', 'BrowserGymStep']
