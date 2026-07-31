"""MiniWoB benchmark evaluated directly through BrowserGym."""

from __future__ import annotations

from typing import Any, Dict

from evalscope.api.agent import NativeAgentConfig
from evalscope.api.benchmark import BenchmarkMeta, BrowserGymAdapter
from evalscope.api.dataset import DatasetDict, Sample, build_dataset_from_records
from evalscope.api.registry import register_benchmark
from evalscope.constants import Tags
from .utils import (
    BROWSER_ACTION_TOOL_INFO,
    MINIWOB_MAX_STEPS,
    MINIWOB_SYSTEM_PROMPT,
    ensure_miniwob_assets,
    load_miniwob_records,
)
from .utils import validate_browser_action as validate_miniwob_action

_DESCRIPTION = """
## Overview

MiniWoB evaluates whether a multimodal agent can complete short browser tasks such as clicking buttons, filling forms,
scrolling and dragging items.

## Task Description

- **Task Type**: Interactive browser tasks
- **Input**: A task goal, an accessibility tree and a screenshot
- **Output**: Browser actions selected through function calling
- **Dataset**: 125 MiniWoB tasks
- **Metrics**: `success_rate` for completed tasks and `error_rate` for environment failures

## Evaluation Notes

- The default run evaluates one deterministic episode per task.
- Set `repeats=5` for the five-episode schedule.
- Each episode allows up to 10 model/tool turns by default.
- The model must support image input and function calling.
- See the [MiniWoB usage guide](../third_party/miniwob.html) for installation and examples.
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
class MiniWobAdapter(BrowserGymAdapter):
    """Run the pinned MiniWoB schedule through BrowserGym."""

    browsergym_module = 'browsergym.miniwob'
    browsergym_action_subset = 'miniwob_all'
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
        """Convert one task record into a BrowserGym episode sample."""
        return Sample(
            input='The task goal and browser observation are supplied when the episode is reset.',
            target='1',
            tools=[BROWSER_ACTION_TOOL_INFO],
            metadata=dict(record),
        )

    def browsergym_task_kwargs(self, sample: Sample) -> Dict[str, Any]:
        """Use the checksum-verified local MiniWoB pages."""
        assets_dir = ensure_miniwob_assets()
        return {'base_url': f'{assets_dir.as_uri()}/'}

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
