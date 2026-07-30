"""Reusable BrowserGym benchmark flow served through OpenEnv."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from PIL import Image

from evalscope.agent.strategies.function_calling import FunctionCallingStrategy
from evalscope.api.agent import ToolExecutionOutput
from evalscope.api.dataset import Sample
from evalscope.api.environment import EnvironmentStepResult, TaskEnvironmentConfig, TaskEnvironmentSession
from evalscope.api.messages import ChatMessage, ChatMessageUser, ContentImage, ContentText
from evalscope.api.tool import ToolCall
from .openenv_adapter import OpenEnvAdapter


class BrowserGymOpenEnvAdapter(OpenEnvAdapter):
    """Shared OpenEnv flow for BrowserGym-backed benchmarks.

    Subclasses declare the BrowserGym benchmark name, provide samples with a
    ``browser_action`` tool, and implement the remaining OpenEnv runtime image
    and dataset hooks. BrowserGym's action transport, reset arguments,
    observations and screenshot artifacts are handled here.
    """

    browsergym_benchmark: str = ''
    browsergym_system_prompt: Optional[str] = None

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        if not self.browsergym_benchmark:
            raise ValueError('browsergym_benchmark must not be empty.')
        self.observation_mode = self.task_environment_config.observation_mode or 'axtree_screenshot'

    def validate_task_environment_config(self, config: TaskEnvironmentConfig) -> None:
        super().validate_task_environment_config(config)
        observation_mode = config.observation_mode or 'axtree_screenshot'
        if observation_mode not in {'axtree', 'axtree_screenshot'}:
            raise ValueError(f'Unsupported BrowserGym observation_mode: {observation_mode}')

    def task_environment_env_vars(self, sample: Sample) -> Dict[str, str]:
        return {
            'BROWSERGYM_BENCHMARK': self.browsergym_benchmark,
            'BROWSERGYM_HEADLESS': 'true',
            'BROWSERGYM_INCLUDE_SCREENSHOT': str(self.observation_mode == 'axtree_screenshot').lower(),
            'MAX_CONCURRENT_ENVS': '1',
        }

    def task_reset_kwargs(self, sample: Sample) -> Dict[str, Any]:
        return {
            'seed': sample.metadata['seed'],
            'task_name': sample.metadata['openenv_task_name'],
        }

    def build_strategy(self, sample: Sample) -> FunctionCallingStrategy:
        return FunctionCallingStrategy(
            system_prompt=self.browsergym_system_prompt,
            include_submit_tool=False,
            max_tool_calls_per_turn=1,
        )

    @staticmethod
    def validate_browser_action(action: str) -> str:
        """Validate and normalize one BrowserGym action expression."""
        action = action.strip()
        if not action:
            raise ValueError('browser_action must not be empty.')
        return action

    def build_task_tools(
        self,
        sample: Sample,
        session: TaskEnvironmentSession,
    ) -> Dict[str, Any]:
        """Bind the BrowserGym action transport to the active OpenEnv session."""

        async def run_browser_action(call: ToolCall, _: Any) -> ToolExecutionOutput:
            action = self.validate_browser_action(str(call.function.arguments.get('action', '')))
            try:
                result = await session.step({'action_str': action})
                observation = self.process_task_result(sample, result, action_performed=True)
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

    def task_observation_messages(
        self,
        sample: Sample,
        observation: Dict[str, Any],
    ) -> List[ChatMessage]:
        if self.observation_mode == 'axtree_screenshot' and observation.get('screenshot') is None:
            raise RuntimeError('OpenEnv did not return a screenshot for axtree_screenshot mode.')
        return [self._observation_message(sample, observation)]

    def normalize_task_observation(self, result: EnvironmentStepResult) -> Dict[str, Any]:
        observation = super().normalize_task_observation(result)
        browsergym_obs = result.metadata.get('browsergym_obs')
        raw_action_error = browsergym_obs.get('last_action_error') if isinstance(browsergym_obs, dict) else None
        if observation.get('last_action_error') and not observation.get('error') and isinstance(raw_action_error, str):
            observation['error'] = raw_action_error
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
        }


__all__ = ['BrowserGymOpenEnvAdapter']
