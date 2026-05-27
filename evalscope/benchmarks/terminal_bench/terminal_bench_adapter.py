import os
from pathlib import Path

from evalscope.api.benchmark import AgentAdapter, BenchmarkMeta
from evalscope.api.dataset import DatasetDict, DictDataLoader, Sample
from evalscope.api.metric import Score
from evalscope.api.model import Model, ModelOutput
from evalscope.api.registry import register_benchmark
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

    def _on_inference(self, model: Model, sample: Sample):
        from harbor.models.trial.config import AgentConfig, EnvironmentConfig
        from harbor.models.trial.config import TaskConfig as TrialTaskConfig
        from harbor.models.trial.config import TrialConfig
        from harbor.trial.trial import Trial

        from .utils import HarborLLM

        environment_config = EnvironmentConfig(type=self.environment_type, **self.environment_kwargs)

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
        return ModelOutput.from_content(
            model=model.name,
            content=result_dict.get('trial_uri', ''),
        )

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
