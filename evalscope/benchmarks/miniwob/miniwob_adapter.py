"""MiniWoB benchmark evaluated through OpenEnv."""

from __future__ import annotations

import numpy as np
import os
import threading
from pathlib import Path
from PIL import Image
from typing import Any, Dict, List, Optional

from evalscope.agent.strategies.function_calling import FunctionCallingStrategy
from evalscope.api.agent import NativeAgentConfig, ToolExecutionOutput
from evalscope.api.benchmark import BenchmarkMeta, OpenEnvAdapter
from evalscope.api.dataset import DatasetDict, Sample, build_dataset_from_records
from evalscope.api.environment import EnvironmentStepResult, TaskEnvironmentConfig, TaskEnvironmentSession
from evalscope.api.messages import ChatMessage, ChatMessageUser, ContentImage, ContentText
from evalscope.api.registry import register_benchmark
from evalscope.api.sandbox import DockerImageSpec, prepare_docker_image
from evalscope.api.tool import ToolCall
from evalscope.constants import Tags
from .utils import (
    BROWSER_ACTION_TOOL_INFO,
    BROWSERGYM_COMMIT,
    BROWSERGYM_METADATA_SHA256,
    BROWSERGYM_VERSION,
    MINIWOB_MAX_STEPS,
    MINIWOB_SYSTEM_PROMPT,
    load_miniwob_records,
    validate_browser_action,
)

OPENENV_VERSION = '0.4.1'
OPENENV_COMMIT = '65c506ef94bb1f7279cb4359673b3ef81031d01f'
OPENENV_PATCH_SHA256 = '465b23aaf7b3b2cadd681495d694a7dad5ca1b36be0cfb5ce5780b94ac354668'
MINIWOB_COMMIT = '7fd85d71a4b60325c6585396ec4f48377d049838'
RUNTIME_PIP_INDEX_URL = 'https://pypi.org/simple'

_IMAGE_BUILD_LOCK = threading.Lock()
_RUNTIME_IMAGE_TAG: Optional[str] = None

_DESCRIPTION = """
MiniWoB evaluates multimodal browser agents on 125 short interactive tasks through OpenEnv and BrowserGym.
The default run uses one deterministic seed per task; set `repeats=5` (or `--repeats 5`) for the full five-seed
schedule. Each episode has a default budget of 10 model/tool turns.

The primary metric is `success_rate`; `error_rate` reports environment failures separately. The default observation
contains both an accessibility tree and a screenshot, so the model must support image input and function calling.

See the [MiniWoB usage guide](../third_party/miniwob.html) for installation, runtime configuration, protocol details
and full-schedule examples.
"""


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
class MiniWobAdapter(OpenEnvAdapter):
    """MiniWoB declarations on top of the reusable OpenEnv episode flow."""

    strategy_name = 'function_calling'
    max_steps_default = MINIWOB_MAX_STEPS

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.observation_mode = self.task_environment_config.observation_mode or 'axtree_screenshot'

    def load(self) -> tuple[DatasetDict, None]:
        """Generate the deterministic task schedule from pinned BrowserGym metadata."""
        records, metadata_path = load_miniwob_records(repeats=self.repeats)
        dataset = build_dataset_from_records(
            records=records,
            sample_fields=self.record_to_sample,
            name='test',
            location=str(metadata_path),
            limit=self.limit,
            repeats=self.repeats,
            shuffle=self.shuffle,
            seed=self.seed,
        )
        for sample in dataset:
            repeat = sample.id % self.repeats
            episode_seeds = sample.metadata.pop('_episode_seeds')
            sample.metadata.update({
                'seed': episode_seeds[repeat],
                'repeat': repeat,
            })
        return DatasetDict({'default': dataset}), None

    def record_to_sample(self, record: Dict[str, Any]) -> Sample:
        """Convert one task record into an OpenEnv episode sample."""
        return Sample(
            input='The task goal and browser observation are supplied when the episode is reset.',
            target='1',
            tools=[BROWSER_ACTION_TOOL_INFO],
            metadata={
                **record,
                'observation_mode': self.observation_mode,
                'openenv_version': OPENENV_VERSION,
                'openenv_commit': OPENENV_COMMIT,
                'openenv_patch_sha256': OPENENV_PATCH_SHA256,
                'browsergym_version': BROWSERGYM_VERSION,
                'browsergym_commit': BROWSERGYM_COMMIT,
                'miniwob_commit': MINIWOB_COMMIT,
                'csv_sha256': BROWSERGYM_METADATA_SHA256,
            },
        )

    def default_task_environment_config(self) -> TaskEnvironmentConfig:
        return TaskEnvironmentConfig.model_validate({
            'backend': 'openenv',
            'observation_mode': 'axtree_screenshot',
            'runtime': {
                'name': 'ms_enclave_docker',
                'config': {},
            },
        })

    def validate_task_environment_config(self, config: TaskEnvironmentConfig) -> None:
        super().validate_task_environment_config(config)
        if config.runtime.name != 'ms_enclave_docker':
            raise ValueError("MiniWoB supports only task_environment.runtime.name='ms_enclave_docker'.")
        observation_mode = config.observation_mode or 'axtree_screenshot'
        if observation_mode not in {'axtree', 'axtree_screenshot'}:
            raise ValueError(f'Unsupported MiniWoB observation_mode: {observation_mode}')

    def prepare_task_environment_image(self) -> str:
        """Build the pinned patched OpenEnv image once per process."""
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

    def task_environment_env_vars(self, sample: Sample) -> Dict[str, str]:
        return {
            'BROWSERGYM_BENCHMARK': 'miniwob',
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
        """Use the standard FC strategy with MiniWoB's one-action contract."""
        return FunctionCallingStrategy(
            system_prompt=MINIWOB_SYSTEM_PROMPT,
            include_submit_tool=False,
            max_tool_calls_per_turn=1,
        )

    def _resolve_strategy(self, sample: Sample, agent_config: Any) -> FunctionCallingStrategy:
        """Keep the benchmark's FC options when the same strategy is explicit."""
        return self.build_strategy(sample)

    def build_task_tools(
        self,
        sample: Sample,
        session: TaskEnvironmentSession,
    ) -> Dict[str, Any]:
        """Bind the MiniWoB action schema to the active OpenEnv session."""

        async def run_browser_action(call: ToolCall, _: Any) -> ToolExecutionOutput:
            action = validate_browser_action(str(call.function.arguments.get('action', '')))
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

    def _validate_agent_config(self) -> None:
        super()._validate_agent_config()
        agent_config = self._task_config.agent_config if self._task_config is not None else None
        if not isinstance(agent_config, NativeAgentConfig):
            return
        if 'strategy' in agent_config.model_fields_set and agent_config.strategy != self.strategy_name:
            raise ValueError("MiniWoB supports only the 'function_calling' strategy.")
        if 'kwargs' in agent_config.model_fields_set:
            raise ValueError('MiniWoB fixes the function_calling strategy options.')
        if agent_config.tools or agent_config.mcp_servers:
            raise ValueError('MiniWoB exposes only its benchmark-owned browser_action tool.')

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


__all__ = ['MiniWobAdapter']
