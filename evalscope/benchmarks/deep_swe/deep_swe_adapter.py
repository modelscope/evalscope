import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from evalscope.api.benchmark import AgentAdapter, BenchmarkMeta
from evalscope.api.benchmark.adapters.dataset_utils import build_dataset_from_records
from evalscope.api.dataset import DatasetDict, Sample
from evalscope.api.evaluator import InferenceResult
from evalscope.api.metric import Score
from evalscope.api.model import Model, ModelOutput
from evalscope.api.registry import register_benchmark
from evalscope.constants import DEFAULT_EVALSCOPE_CACHE_DIR, Tags
from evalscope.utils.function_utils import AsyncioLoopRunner
from evalscope.utils.import_utils import check_import
from evalscope.utils.logger import get_logger
from .utils import (
    build_score_metadata,
    download_snapshot,
    filter_task_records,
    first_trial_result,
    load_pier_trace,
    load_task_records,
)

logger = get_logger()

DEFAULT_MODELSCOPE_DATASET_ID = 'evalscope/deep-swe'

COMMON_EXTRA_PARAMS = {
    'task_ids': {
        'type': 'list',
        'description': 'Optional list of DeepSWE task ids to evaluate.',
        'value': [],
    },
    'languages': {
        'type': 'list',
        'description': 'Optional task language filter from manifest metadata.',
        'value': [],
    },
    'categories': {
        'type': 'list',
        'description': 'Optional task category filter from manifest metadata.',
        'value': [],
    },
    'sample_seed': {
        'type': 'int',
        'description': 'Optional deterministic shuffle seed applied before limit.',
        'value': '',
    },
    'pier_agent_kwargs': {
        'type': 'dict',
        'description': 'Extra kwargs passed to Pier AgentConfig.kwargs.',
        'value': {},
    },
}


class DeepSWEAdapter(AgentAdapter):
    """EvalScope adapter for DeepSWE through Pier Python API jobs."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        extra_params = self.extra_params or {}
        self.task_ids = self._as_list(extra_params.get('task_ids') or [])
        self.languages = self._as_list(extra_params.get('languages') or [])
        self.categories = self._as_list(extra_params.get('categories') or [])
        self.sample_seed = extra_params.get('sample_seed')
        self.pier_agent_kwargs = dict(extra_params.get('pier_agent_kwargs') or {})

    @staticmethod
    def _as_list(value: Union[str, List[Any], Tuple[Any, ...]]) -> List[str]:
        if isinstance(value, str):
            return [value] if value else []
        return [str(item) for item in value]

    def load(self) -> Tuple[DatasetDict, None]:
        snapshot_path = self._download_snapshot()
        task_records = filter_task_records(
            load_task_records(snapshot_path),
            task_ids=self.task_ids,
            languages=self.languages,
            categories=self.categories,
        )
        dataset = build_dataset_from_records(
            records=task_records,
            sample_fields=self.record_to_sample,
            name=self.eval_split,
            location=str(snapshot_path),
            limit=self.limit,
            repeats=self.repeats,
            shuffle=self.shuffle or self.sample_seed not in (None, ''),
            seed=self._sample_seed(),
        )
        return DatasetDict({self.eval_split: dataset}), None

    def record_to_sample(self, record: Dict[str, Any]) -> Sample:
        return Sample(input=record.get('instruction', ''), target='', metadata=record)

    def _download_snapshot(self) -> Path:
        cache_dir = Path(DEFAULT_EVALSCOPE_CACHE_DIR) / self.name / 'snapshots'
        logger.info(f'Loading DeepSWE snapshot from {self.dataset_hub}: {self.dataset_id}')
        return download_snapshot(
            data_id_or_path=self.dataset_id,
            data_source=self.dataset_hub,
            force_redownload=self.force_redownload,
            cache_dir=str(cache_dir),
        )

    def _sample_seed(self) -> Optional[int]:
        if self.sample_seed in (None, ''):
            return self.seed
        return int(self.sample_seed)

    def _on_inference(self, model: Model, sample: Sample) -> InferenceResult:
        result_dict = self._run_pier_job(model, sample)
        sample.metadata['result'] = result_dict

        trial_uri = first_trial_result(result_dict).get('trial_uri') or result_dict.get('trial_uri', '')
        output = ModelOutput.from_content(model=model.name, content=trial_uri)
        trace, messages = load_pier_trace(result_dict)
        return InferenceResult(output=output, trace=trace, messages=messages)

    def _run_pier_job(self, model: Model, sample: Sample) -> Dict[str, Any]:
        check_import('pier', extra='deep_swe', raise_error=True, feature_name=self.pretty_name)

        from pier.job import Job
        from pier.models.job.config import JobConfig
        from pier.models.trial.config import AgentConfig, EnvironmentConfig, TaskConfig, VerifierConfig

        task_id = str(sample.metadata['task_id'])

        config = JobConfig(
            job_name=f'{task_id[:48].rstrip("_-")}__{uuid.uuid4().hex[:8]}',
            jobs_dir=Path(self.output_dir) / 'deep_swe_jobs',
            n_attempts=1,
            n_concurrent_trials=1,
            quiet=True,
            timeout_multiplier=1.0,
            agent_timeout_multiplier=1.0,
            verifier_timeout_multiplier=1.0,
            environment_build_timeout_multiplier=1.0,
            agents=[AgentConfig(
                name='mini-swe-agent',
                model_name=model.name,
                kwargs=self.pier_agent_kwargs,
                env={},
            )],
            environment=EnvironmentConfig(type='docker'),
            verifier=VerifierConfig(env={}),
            tasks=[TaskConfig(path=Path(sample.metadata['task_path']))],
        )

        async def _run_job() -> Any:
            job = await Job.create(config)
            return await job.run()

        result = AsyncioLoopRunner.run(_run_job())
        result_dict = result.model_dump(mode='json')
        result_dict['job_result_path'] = str(Path(config.jobs_dir) / config.job_name)
        self._raise_for_pier_failures(result_dict)
        return result_dict

    @staticmethod
    def _raise_for_pier_failures(result_dict: Dict[str, Any]) -> None:
        trial_results = result_dict.get('trial_results') or []
        if not trial_results:
            raise RuntimeError('Pier DeepSWE job did not return any trial results.')

        trial_result = trial_results[0]
        rewards = ((trial_result.get('verifier_result') or {}).get('rewards') or {})
        if rewards.get('reward') is not None:
            return

        exception_info = trial_result.get('exception_info')
        if exception_info:
            exc_type = exception_info.get('exception_type') or exception_info.get('type') or 'UnknownPierError'
            exc_msg = exception_info.get('message') or exception_info.get('exception_message') or str(exception_info)
            raise RuntimeError(f'Pier DeepSWE trial failed with {exc_type}: {exc_msg}')
        raise RuntimeError('Pier DeepSWE trial did not return a reward or exception info.')

    def match_score(self, original_prediction: str, filtered_prediction: str, reference: str, task_state: Any) -> Score:
        result = task_state.metadata.get('result', {})
        metadata = build_score_metadata(result)
        reward = metadata.get('reward')
        acc = float(reward if reward is not None else 0.0)
        return Score(
            extracted_prediction=filtered_prediction,
            prediction=original_prediction,
            value={'acc': acc},
            metadata=metadata,
        )


@register_benchmark(
    BenchmarkMeta(
        name='deep_swe',
        pretty_name='DeepSWE',
        tags=[Tags.CODING, Tags.AGENT, Tags.MULTI_TURN],
        description="""
## Overview

DeepSWE is a coding-agent benchmark for evaluating repository-level software engineering tasks. EvalScope
integrates it through Pier and runs each benchmark sample as one Pier Python API job.

## Task Description

- **Task Type**: Agentic software engineering
- **Input**: DeepSWE task directory containing task metadata and verifier assets
- **Output**: A repository patch produced by a Pier built-in agent
- **Scoring**: Binary verifier reward exposed as `acc`

## Evaluation Notes

- Requires **Python>=3.12**, Docker, and `pip install evalscope[deep_swe]`
- Dataset defaults to ModelScope `evalscope/deep-swe`
- DeepSWE runs through Pier's Docker environment in EvalScope
- Use `pier_agent_kwargs={'model_class': 'litellm'}` for OpenAI-compatible providers that do not support Responses API
""",
        dataset_id=DEFAULT_MODELSCOPE_DATASET_ID,
        eval_split='test',
        prompt_template='{question}',
        metric_list=['acc'],
        extra_params=COMMON_EXTRA_PARAMS,
    )
)
class DeepSWEBenchmarkAdapter(DeepSWEAdapter):
    pass
