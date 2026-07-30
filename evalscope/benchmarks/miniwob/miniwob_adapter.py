"""MiniWoB benchmark evaluated through OpenEnv."""

from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Any, Dict, Optional

from evalscope.api.agent import NativeAgentConfig
from evalscope.api.benchmark import BenchmarkMeta, BrowserGymOpenEnvAdapter
from evalscope.api.dataset import DatasetDict, Sample, build_dataset_from_records
from evalscope.api.environment import TaskEnvironmentConfig
from evalscope.api.registry import register_benchmark
from evalscope.api.sandbox import DockerImageSpec, prepare_docker_image
from evalscope.constants import Tags
from .utils import (
    BROWSER_ACTION_TOOL_INFO,
    BROWSERGYM_COMMIT,
    BROWSERGYM_METADATA_SHA256,
    BROWSERGYM_VERSION,
    MINIWOB_MAX_STEPS,
    MINIWOB_SYSTEM_PROMPT,
    load_miniwob_records,
)
from .utils import validate_browser_action as validate_miniwob_action

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
class MiniWobAdapter(BrowserGymOpenEnvAdapter):
    """MiniWoB declarations on top of the reusable OpenEnv episode flow."""

    browsergym_benchmark = 'miniwob'
    browsergym_system_prompt = MINIWOB_SYSTEM_PROMPT
    strategy_name = 'function_calling'
    max_steps_default = MINIWOB_MAX_STEPS
    validate_browser_action = staticmethod(validate_miniwob_action)

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


__all__ = ['MiniWobAdapter']
