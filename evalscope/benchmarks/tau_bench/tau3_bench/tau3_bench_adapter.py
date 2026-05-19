import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict

from evalscope.api.benchmark import AgentAdapter, BenchmarkMeta
from evalscope.api.dataset import Sample
from evalscope.api.dataset.dataset import DatasetDict
from evalscope.api.dataset.loader import DictDataLoader
from evalscope.api.evaluator import InferenceResult
from evalscope.api.messages.chat_message import ChatMessageUser
from evalscope.api.metric import Score
from evalscope.api.model import Model
from evalscope.api.registry import register_benchmark
from evalscope.constants import Tags
from evalscope.utils import get_logger
from evalscope.utils.import_utils import check_import

logger = get_logger()


@register_benchmark(
    BenchmarkMeta(
        name='tau3_bench',
        pretty_name='τ³-bench',
        tags=[Tags.FUNCTION_CALLING, Tags.REASONING, Tags.AGENT],
        description="""
## Overview

τ³-bench (Tau Cubed Bench) is the v1.0.0 release of the tau-bench family. It extends τ²-bench with a knowledge-retrieval domain, voice/audio-native evaluation, and 75+ task fixes across the existing domains.

## Task Description

- **Task Type**: Conversational Agent Evaluation with optional knowledge retrieval
- **Input**: User scenarios with complex goals and multi-step requirements
- **Output**: Agent actions via API tool calls following policy guidelines
- **Domains**: Airline, Retail, Telecom, Banking Knowledge

## Key Features

- New `banking_knowledge` domain with 97 tasks and 698 policy/procedure documents (RAG)
- 75+ task quality fixes across airline / retail / banking domains
- Pluggable retrieval pipeline: BM25, dense embeddings (OpenAI / Qwen), grep, sandbox shell, rerankers
- LLM-simulated user interactions, multi-turn dialogue with tool calling

## Evaluation Notes

- **Python**: requires 3.12-3.13
- **Installation Required**: `pip install 'tau2[knowledge] @ git+https://github.com/sierra-research/tau2-bench@v1.0.0'`
- **Cannot coexist with `tau2_bench`** in the same environment (same PyPI package name `tau2`, different versions). Pick one.
- **User Model Configuration**: Requires setting up a user simulation model
- **Retrieval config (banking_knowledge only)**: defaults to `bm25` (offline). Switch via `extra_params.retrieval_config`. Other configs may need extra deps:
  - `bm25` → ships with `[knowledge]` extra (no API key)
  - `openai_embeddings*` → set `OPENAI_API_KEY`
  - `qwen_embeddings*` → set `OPENROUTER_API_KEY`
  - `*_reranker` → also needs `OPENAI_API_KEY`
  - `terminal_use` / `alltools*` → require Anthropic `sandbox-runtime` (npm) + ripgrep / bwrap / socat (see tau2 README)
- Primary metric: **Accuracy** based on task completion reward
- Uses **pass@k** aggregation for robustness evaluation
- [Usage Example](https://evalscope.readthedocs.io/en/latest/third_party/tau3_bench.html)
""",  # noqa: E501
        dataset_id='evalscope/tau3-bench-data',
        subset_list=['airline', 'retail', 'telecom', 'banking_knowledge'],
        aggregation='mean_and_pass_hat_k',
        eval_split='test',
        extra_params={
            'user_model': {
                'type': 'str',
                'description': 'Model used to simulate the user in the environment.',
                'value': 'qwen-plus'
            },
            'api_key': {
                'type': 'str',
                'description': 'API key for the user model backend.',
                'value': 'EMPTY'
            },
            'api_base': {
                'type': 'str',
                'description': 'Base URL for the user model API requests.',
                'value': 'https://dashscope.aliyuncs.com/compatible-mode/v1'
            },
            'generation_config': {
                'type': 'dict',
                'description': 'Default generation config for user model simulation.',
                'value': {
                    'temperature': 0.0,
                }
            },
            'retrieval_config': {
                'type': 'str',
                'description': (
                    'Retrieval config name for the banking_knowledge domain. '
                    'Common values: no_knowledge, full_kb, golden_retrieval, bm25, '
                    'openai_embeddings, qwen_embeddings, *_reranker, *_grep, '
                    'terminal_use, alltools. Ignored for non-knowledge domains.'
                ),
                'value': 'bm25'
            },
            'retrieval_config_kwargs': {
                'type': 'dict',
                'description': 'Optional kwargs forwarded to the retrieval pipeline.',
                'value': {}
            }
        }
    )
)
class Tau3BenchAdapter(AgentAdapter):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Resolve dataset path and set TAU2_DATA_DIR BEFORE importing tau2.
        # tau2 v1.0.0 caches DATA_DIR at module import time; setting the env var
        # later has no effect.
        self._prepare_data_dir()

        check_import(
            'tau2',
            package="'tau2[knowledge] @ git+https://github.com/sierra-research/tau2-bench@v1.0.0'",
            raise_error=True,
            feature_name=self.pretty_name
        )

        # setup user model args
        self.user_model = self.extra_params.get('user_model', 'qwen-plus')
        self.api_key = self.extra_params.get('api_key', 'EMPTY')
        self.api_base = self.extra_params.get('api_base', 'https://dashscope.aliyuncs.com/compatible-mode/v1')
        self.generation_config = self.extra_params.get('generation_config', {'temperature': 0.0, 'max_tokens': 4096})

        # retrieval config (banking_knowledge only)
        self.retrieval_config = self.extra_params.get('retrieval_config', 'bm25')
        self.retrieval_config_kwargs = self.extra_params.get('retrieval_config_kwargs', {}) or {}

    def _prepare_data_dir(self):
        dataset_name_or_path = self.dataset_id
        if os.path.exists(dataset_name_or_path):
            logger.info(f'Loading dataset from {dataset_name_or_path}')
            dataset_path = dataset_name_or_path
        else:
            from modelscope import dataset_snapshot_download
            logger.info(f'Loading dataset from modelscope: > dataset_name: {dataset_name_or_path}')
            dataset_path = dataset_snapshot_download(dataset_name_or_path)
        os.environ['TAU2_DATA_DIR'] = dataset_path
        self._dataset_path = dataset_path
        # If tau2 was imported earlier in this process (e.g. by tau2_bench in
        # multi-benchmark workflows like `make docs`), its DATA_DIR constant is
        # frozen to the fallback path. Patch it so domain modules loaded later
        # resolve files under our downloaded dataset.
        tau2_utils = sys.modules.get('tau2.utils.utils')
        if tau2_utils is not None:
            tau2_utils.DATA_DIR = Path(dataset_path)

    def load(self):
        # Data directory and env var were prepared in __init__.
        from tau2.registry import registry

        data_dict = defaultdict(dict)
        for domain_name in self.subset_list:
            logger.info(f'Loading Tau3-Bench environment: {domain_name}')
            task_loader = registry.get_tasks_loader(domain_name)
            tasks = task_loader()
            tasks = [task.model_dump(exclude_unset=True) for task in tasks]

            # Inject the domain name so record_to_sample can use it as subset_key.
            # Different domains have different task schemas (e.g. banking_knowledge
            # uses a string for user_scenario.instructions, while airline uses a dict),
            # so the loop-level domain_name is the only reliable source.
            for t in tasks:
                t['_domain'] = domain_name

            dataset = DictDataLoader(
                dict_list=tasks,
                sample_fields=self.record_to_sample,
                limit=self.limit,
                repeats=self.repeats,
                shuffle=self.shuffle,
            ).load()

            data_dict[domain_name] = dataset

        test_dataset = DatasetDict(data_dict)

        return test_dataset, None

    def record_to_sample(self, record: Dict) -> Sample:
        """Convert a data record to a Sample object."""
        purpose = record.get('description', {}).get('purpose') or ''
        return Sample(
            input=[ChatMessageUser(content=purpose)],
            target='',  # Will use the record for evaluation
            subset_key=record['_domain'],
            metadata=record  # Store the full record for evaluation
        )

    def _on_inference(self, model: Model, sample: Sample) -> InferenceResult:
        from .generation import predict
        return predict(model, sample, adapter_instance=self)

    def match_score(self, original_prediction: str, filtered_prediction: str, reference: str, task_state) -> Score:

        score = Score(
            extracted_prediction=filtered_prediction,
            prediction=original_prediction,
        )

        try:
            task_result = task_state.metadata['task_result']
            reward = task_result['reward']

            score.value = {
                'acc': float(reward),
            }
            score.explanation = f'Task completed with reward: {reward}'
            score.metadata = {
                'task_result': task_result,
            }
            score.main_score_name = 'acc'

        except Exception as e:
            score.value = {'acc': 0.0}
            score.explanation = f'Evaluation failed: {str(e)}'
            score.metadata = {'error': str(e)}
            score.main_score_name = 'acc'

        return score
