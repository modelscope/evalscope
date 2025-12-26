import asyncio
import os
from pathlib import Path
from uuid import uuid4

from evalscope.api.benchmark import BenchmarkMeta, DefaultDataAdapter
from evalscope.api.dataset import DatasetDict, DictDataLoader, Sample
from evalscope.api.evaluator import TaskState
from evalscope.api.metric import Score
from evalscope.api.model import Model, ModelOutput
from evalscope.api.registry import register_benchmark
from evalscope.constants import DEFAULT_EVALSCOPE_CACHE_DIR, Tags
from evalscope.utils.import_utils import check_import
from evalscope.utils.logger import get_logger

logger = get_logger()


@register_benchmark(
    BenchmarkMeta(
        name='terminal_bench_v2',
        pretty_name='Terminal-Bench-2.0',
        tags=[Tags.CODING],
        description='Terminal-Bench v2 is a command-line benchmark suite that evaluates AI agents on 89 real-world, '
        'multi-step terminal tasks—ranging from compiling and debugging to system administration—within '
        'isolated containers; each task is rigorously validated and auto-scored 0/1, pushing frontier '
        'models to prove they can act, not just answer. '
        'Require `Python>=3.12` and need to run `pip install harbor==0.1.28` before evaluating. '
        '[Usage Example](https://evalscope.readthedocs.io/en/latest/third_party/terminal_bench.html)',
        dataset_id='https://github.com/laude-institute/terminal-bench-2.git',
        metric_list=['acc'],
        eval_split='test',
        prompt_template='{question}',
        extra_params={
            'environment_type': {
                'type': 'str',
                'description': 'Environment type for running the benchmark.',
                'value': 'docker',
                'choices': ['docker', 'daytona', 'e2b', 'modal']
            },
            'agent_name': {
                'type':
                'str',
                'description':
                'Agent type to be used in Harbor.',
                'value':
                'terminus-2',
                'choices': [
                    'oracle', 'terminus-2', 'claude-code', 'codex', 'qwen-coder', 'openhands', 'opencode',
                    'mini-swe-agent'
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
        }
    )
)
class TerminalBenchV2Adapter(DefaultDataAdapter):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        check_import('harbor', package='harbor==0.1.28', raise_error=True, feature_name=self.pretty_name)
        self.environment_type = self.extra_params.get('environment_type', 'docker')
        self.agent_name = self.extra_params.get('agent_name', 'terminus-2')
        self.timeout_multiplier = self.extra_params.get('timeout_multiplier', 1.0)
        self.max_turns = self.extra_params.get('max_turns', 200)

    def load(self):
        from harbor.dataset.client import DatasetClient
        from harbor.models.job.config import RegistryDatasetConfig
        from harbor.models.registry import RemoteRegistryInfo

        config = RegistryDatasetConfig(
            registry=RemoteRegistryInfo(),
            name='terminal-bench',
            version='2.0',
            overwrite=self.force_redownload,
            download_dir=os.path.join(DEFAULT_EVALSCOPE_CACHE_DIR, self.name)
        )
        client = DatasetClient()

        logger.info(f'Downloading dataset for {self.pretty_name}...')
        _ = client.download_dataset_from_config(config)
        task_config = config.get_task_configs()

        datasets = {}
        dataset = DictDataLoader(
            dict_list=[task.model_dump(mode='json') for task in task_config],
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
        from harbor.models.agent.name import AgentName
        from harbor.models.trial.config import AgentConfig, EnvironmentConfig, TaskConfig, TrialConfig
        from harbor.orchestrators.factory import OrchestratorFactory

        from .utils import HarborLLM

        harbor_llm = HarborLLM(model=model)
        environment_config = EnvironmentConfig(type=self.environment_type, )

        agent_config = AgentConfig(
            name=self.agent_name,
            model_name=model.name,
            kwargs={
                # Parser configuration
                'parser_name': 'json',  # "json" or "xml" (default: "json")
                # model
                'llm': harbor_llm,
                # Episode/turn limits
                'max_turns': self.max_turns,  # Maximum number of episodes (default: 1000000)
                # Summarization configuration
                'enable_summarize': True,  # Enable context summarization (default: True)
                'proactive_summarization_threshold': 8000,  # Free tokens threshold for summarization (default: 8000)
                # RL training configuration (default: False)
                # If enabled, token ids and logprobs are collected in result and persisted in trajectories
                'collect_rollout_details': False,
            }
        )

        task_config = TaskConfig.model_validate(sample.metadata)
        trial_config = TrialConfig(
            task=task_config,
            trials_dir=Path(self.output_dir) / 'trials',
            agent=agent_config,
            environment=environment_config,
            timeout_multiplier=self.timeout_multiplier,
            job_id=uuid4(),
        )

        orchestrator = OrchestratorFactory.create_orchestrator(
            orchestrator_type='local',
            trial_configs=[trial_config],
            n_concurrent_trials=1,
            metrics=None,
        )

        try:
            result = asyncio.run(orchestrator.run())
        except Exception as e:
            if hasattr(e, 'exceptions'):
                for i, sub_exc in enumerate(e.exceptions):
                    logger.warning(f'--- Sub-exception {i + 1} ---')
                    logger.warning(sub_exc)
            else:
                logger.warning(e)
            raise e
        result_dict = result[0].model_dump(mode='json')

        sample.metadata['result'] = result_dict
        return ModelOutput.from_content(
            model=model.name,
            content=result_dict['trial_uri'],
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
