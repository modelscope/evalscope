"""GAIA benchmark adapter.

Drives a multi-turn :class:`AgentLoop` per sample with the built-in
``react`` strategy and a single ``bash`` tool inside a Docker sandbox.
Per-sample attachment files (PDF / xlsx / images / ...) are exposed to the
agent via a read-only volume mount of the dataset snapshot ``2023/<split>/``
directory at ``/shared_files``.

Mirrors the inspect_ai GAIA implementation
(https://github.com/UKGovernmentBEIS/inspect_evals/blob/main/src/inspect_evals/gaia/)
for prompt template, attachment exposure path and scorer; deviates by:

* Using only ``bash`` (no ``python`` or ``web_browser`` tool).
* Using ``python:3.11-slim`` instead of ``aisiuk/inspect-tool-support``.
* Mounting the whole split directory read-only instead of per-sample
  ``sample.files`` copy (evalscope's ``Sample.files`` field is currently
  not consumed by any environment).
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

from evalscope.agent.tools.bash import BASH_TOOL_INFO, run_bash
from evalscope.api.agent import AgentEnvironment, AgentStrategy
from evalscope.api.benchmark import BenchmarkMeta
from evalscope.api.benchmark.adapters import AgentLoopAdapter
from evalscope.api.dataset import Sample
from evalscope.api.evaluator import TaskState
from evalscope.api.metric import Score
from evalscope.api.registry import get_strategy, register_benchmark
from evalscope.constants import HubType, Tags
from evalscope.utils.import_utils import is_build_doc
from evalscope.utils.logger import get_logger
from .scorer import question_scorer

logger = get_logger()

# ---------------------------------------------------------------------------
# Prompt — verbatim from inspect_evals/gaia/dataset.py (DEFAULT_INPUT_PROMPT)
# ---------------------------------------------------------------------------

DEFAULT_INPUT_PROMPT = """Please answer the question below. You should:

- Return only your answer, which should be a number, or a short phrase with as few words as possible, or a comma separated list of numbers and/or strings.
- If the answer is a number, return only the number without any units unless specified otherwise.
- If the answer is a string, don't include articles, and don't use abbreviations (e.g. for states).
- If the answer is a comma separated list, apply the above rules to each element in the list.

{file}

Here is the question:

{question}"""

_DEFAULT_DOCKER_IMAGE = 'python:3.11'
_SHARED_FILES_DIR = '/shared_files'
_GAIA_DATASET_ID = 'gaia-benchmark/GAIA'

_GAIA_EXTRA_PARAMS: Dict[str, Any] = {
    'max_steps': {
        'type': 'int',
        'description': 'Maximum number of agent steps per sample.',
        'value': 50,
    },
    'command_timeout': {
        'type': 'float',
        'description': 'Default per-bash-command timeout in seconds.',
        'value': 180.0,
    },
    'docker_image': {
        'type': 'str',
        'description': 'Docker image used as the per-sample sandbox.',
        'value': _DEFAULT_DOCKER_IMAGE,
    },
    'network_enabled': {
        'type': 'bool',
        'description': 'Allow the sandbox to access the network (GAIA browsing questions need this).',
        'value': True,
    },
}


@register_benchmark(
    BenchmarkMeta(
        name='gaia',
        pretty_name='GAIA',
        tags=[Tags.AGENT, Tags.REASONING, Tags.MULTI_TURN],
        description="""
## Overview

GAIA (General AI Assistants) is a benchmark of 450+ questions targeting next-generation LLMs with tool use, web browsing and multi-step reasoning. Each question has an unambiguous short answer and is bucketed into one of three difficulty levels.

## Task Description

- **Task Type**: Tool-use Agent (multi-turn)
- **Input**: Natural-language question, optionally with one referenced attachment file (PDF / xlsx / image / audio / ...)
- **Output**: A short final answer (number / short phrase / comma-separated list)
- **Splits**: ``validation`` (answers public) and ``test`` (answers private — evaluation skipped)

## Key Features

- ReAct agent loop with a single ``bash`` tool inside a Docker sandbox (default image ``python:3.11``, includes ``curl`` / ``wget`` / ``git``).
- Attachment files are mounted read-only at ``/shared_files`` inside the sandbox.
- Rule-based scorer ported verbatim from the official GAIA leaderboard (no LLM judge).
- Dataset downloaded from ModelScope (``gaia-benchmark/GAIA``) by default; set ``dataset_hub='huggingface'`` to load from Hugging Face instead.

## Evaluation Notes

- Requires Docker daemon running locally (or a remote sandbox engine via the ms_enclave configuration).
- ``extra_params.max_steps`` caps the agent loop length (default 50).
- ``extra_params.command_timeout`` sets per-``bash`` command timeout (default 180s, mirrors inspect_ai).
- Network is enabled by default — many questions require browsing/searching.
- Use ``subset_list`` to restrict to specific difficulty levels, e.g. ``['2023_level1']``, ``['2023_level1', '2023_level2']`` or ``['2023_all']`` (default).
- [Usage Documentation](https://evalscope.readthedocs.io/en/latest/third_party/gaia.html)
""",
        dataset_id=_GAIA_DATASET_ID,
        metric_list=['acc'],
        eval_split='validation',
        subset_list=['2023_level1', '2023_level2', '2023_level3'],
        default_subset='2023_level1',
        prompt_template='{question}',
        extra_params=_GAIA_EXTRA_PARAMS,
    )
)
class GaiaAdapter(AgentLoopAdapter):
    """GAIA adapter: ReAct + bash, Docker sandbox, dataset attachments mounted at /shared_files."""

    strategy_name: str = 'react'
    max_steps_default: int = 50

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

        self.command_timeout: float = float(self.extra_params.get('command_timeout', 180.0))
        self.docker_image: str = self.extra_params.get('docker_image', _DEFAULT_DOCKER_IMAGE)
        self.network_enabled: bool = bool(self.extra_params.get('network_enabled', True))

        self._snapshot_dir: Optional[str] = None
        self._host_files_dir: Optional[str] = None

    # ------------------------------------------------------------------
    # Dataset
    # ------------------------------------------------------------------

    def record_to_sample(self, record: Dict[str, Any]) -> Sample:
        file_name = record.get('file_name') or ''
        if file_name:
            file_hint = (
                'The following file is referenced in the question below and you will likely need to use it in '
                f'order to find the correct answer: {_SHARED_FILES_DIR}/{file_name}'
            )
        else:
            file_hint = ''

        input_text = DEFAULT_INPUT_PROMPT.format(file=file_hint, question=record['Question'])

        return Sample(
            input=input_text,
            target=str(record.get('Final answer', '') or ''),
            metadata={
                'task_id': record['task_id'],
                'level': record.get('Level'),
                'file_name': file_name,
                'file_path': record.get('file_path') or '',
                'Annotator Metadata': record.get('Annotator Metadata'),
            },
        )

    def _post_process_samples(self) -> None:
        """Pull the full GAIA snapshot once, then attach the bash tool to every sample."""
        self._ensure_snapshot()

        for subset_samples in self.test_dataset.values():
            for sample in subset_samples:
                existing_tools = list(sample.tools or [])
                if not any(t.name == 'bash' for t in existing_tools):
                    existing_tools.append(BASH_TOOL_INFO)
                sample.tools = existing_tools

        super()._post_process_samples()

    def _ensure_snapshot(self) -> None:
        """Download the GAIA repo snapshot so attachment files are available for mounting."""
        if self._snapshot_dir is not None or is_build_doc():
            return

        if self.dataset_hub == HubType.LOCAL or os.path.exists(self.dataset_id):
            self._snapshot_dir = os.path.abspath(self.dataset_id)
        elif self.dataset_hub == HubType.HUGGINGFACE:
            from huggingface_hub import snapshot_download

            logger.info(f'Downloading GAIA snapshot from Hugging Face (split={self.eval_split})...')
            self._snapshot_dir = snapshot_download(
                repo_id=self.dataset_id,
                repo_type='dataset',
                allow_patterns=[f'2023/{self.eval_split}/*'],
            )
        else:
            from modelscope import dataset_snapshot_download

            logger.info(f'Downloading GAIA snapshot from ModelScope (split={self.eval_split})...')
            self._snapshot_dir = dataset_snapshot_download(
                self.dataset_id,
                allow_file_pattern=[f'2023/{self.eval_split}/*'],
            )

        split_dir = os.path.join(self._snapshot_dir, '2023', self.eval_split)
        if not os.path.isdir(split_dir):
            logger.warning(
                f'GAIA attachment directory {split_dir} missing; '
                'samples with file_name will not be able to read their files.'
            )
        self._host_files_dir = split_dir

    # ------------------------------------------------------------------
    # AgentLoop hooks
    # ------------------------------------------------------------------

    def build_strategy(self, sample: Sample) -> AgentStrategy:
        strategy_cls = get_strategy(self.strategy_name)
        return strategy_cls()

    def build_tools(self, sample: Sample) -> Dict[str, Any]:
        return {'bash': run_bash}

    def build_environment(self, sample: Sample) -> Optional[AgentEnvironment]:
        from evalscope.agent.environments.enclave import EnclaveAgentEnvironment

        volumes: Dict[str, Dict[str, str]] = {}
        if self._host_files_dir and os.path.isdir(self._host_files_dir):
            volumes[self._host_files_dir] = {'bind': _SHARED_FILES_DIR, 'mode': 'ro'}

        sandbox_config: Dict[str, Any] = {
            'image': self.docker_image,
            'working_dir': '/workspace',
            'network_enabled': self.network_enabled,
            'environment': {
                'PAGER': 'cat',
                'MANPAGER': 'cat',
                'PIP_PROGRESS_BAR': 'off',
                'TQDM_DISABLE': '1',
            },
        }
        if volumes:
            sandbox_config['volumes'] = volumes

        return EnclaveAgentEnvironment(
            engine='docker',
            sandbox_config=sandbox_config,
            timeout=self.command_timeout,
        )

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def match_score(
        self,
        original_prediction: str,
        filtered_prediction: str,
        reference: str,
        task_state: TaskState,
    ) -> Score:
        answer = filtered_prediction or ''
        target = reference or ''
        ok, explanation = question_scorer(model_answer=answer, ground_truth=target)
        return Score(
            extracted_prediction=answer,
            prediction=original_prediction,
            value={'acc': 1.0 if ok else 0.0},
            explanation=explanation,
            metadata={'target': target},
        )
