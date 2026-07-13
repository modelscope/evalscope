# Copyright (c) Alibaba, Inc. and its affiliates.
# Scoring logic adapted from https://github.com/databricks/officeqa/blob/main/reward.py

import os
from typing import Any, Dict, List, Optional

from evalscope.agent.tools.bash import BASH_TOOL_INFO, run_bash
from evalscope.api.agent import AgentEnvironment
from evalscope.api.benchmark import BenchmarkMeta
from evalscope.api.benchmark.adapters import AgentLoopAdapter
from evalscope.api.dataset import Sample, resolve_snapshot_or_local_path
from evalscope.api.evaluator import TaskState
from evalscope.api.messages import ChatMessageUser
from evalscope.api.metric import Score
from evalscope.api.registry import register_benchmark
from evalscope.constants import Tags
from evalscope.utils.logger import get_logger
from .utils import DEFAULT_TOLERANCE, extract_final_answer, score_answer

logger = get_logger()

DESCRIPTION = """
## Overview

OfficeQA is a grounded reasoning benchmark by Databricks, built for evaluating model/agent performance on end-to-end grounded reasoning tasks over U.S. Treasury Bulletin documents (1939-2025).

## Task Description

- **Task Type**: Agent-based Document QA (grep/search over corpus)
- **Input**: A question + access to parsed Treasury Bulletin text files via bash tools
- **Output**: A precise answer (numeric values, text, or structured data)
- **Evaluation Mode**: Agent with bash tool (grep, cat, etc.) over the corpus

## Key Features

- Two subsets: `officeqa_pro` (133 questions, hard, default) and `officeqa_full` (246 questions, easy+hard)
- Corpus: ~900 parsed Treasury Bulletin text files (~460MB total)
- Agent uses bash tools (grep, cat, head, etc.) to search the corpus
- Scoring uses fuzzy numeric matching with configurable tolerance (1% default)

## Evaluation Notes

- The agent is given access to parsed .txt files in a corpus directory
- Each question's `source_files` field indicates which document(s) contain the answer
- Uses **rule-based scoring** adapted from official reward.py
- Numerical answers matched with 1% relative error tolerance
- Text answers use case-insensitive substring matching
"""

SYSTEM_PROMPT = """\
You are an expert research assistant with access to a corpus of U.S. Treasury Bulletin documents.
Use the bash tool to search through the text files and find precise answers to questions.

The corpus is located at: {corpus_dir}
Each file is named like treasury_bulletin_YYYY_MM.txt (e.g., treasury_bulletin_1941_01.txt).

Tips:
- Use `grep -r "keyword" {corpus_dir}/` to search across files
- Use `cat {corpus_dir}/filename.txt` to read a specific file
- Use `grep -n "pattern" file` to find line numbers
- Answers should be precise numbers or short text. Wrap your final answer in <FINAL_ANSWER>...</FINAL_ANSWER> tags.
"""

INPUT_TEMPLATE = """\
Question: {question}

The relevant document(s): {source_files}

Search the corpus to find the answer. Provide ONLY the final answer wrapped in <FINAL_ANSWER>...</FINAL_ANSWER> tags.
"""

# Container-internal path when using Docker sandbox
_CONTAINER_CORPUS_DIR = '/corpus'


@register_benchmark(
    BenchmarkMeta(
        name='officeqa',
        pretty_name='OfficeQA',
        dataset_id='evalscope/officeqa',
        tags=[Tags.AGENT, Tags.QA, Tags.KNOWLEDGE],
        description=DESCRIPTION,
        subset_list=['officeqa_pro'],
        default_subset='officeqa_pro',
        metric_list=['acc'],
        few_shot_num=0,
        eval_split='train',
        prompt_template='{question}',
    )
)
class OfficeQAAdapter(AgentLoopAdapter):

    strategy_name: str = 'function_calling'
    max_steps_default: int = 15

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._corpus_dir: Optional[str] = None

    # ------------------------------------------------------------------
    # Dataset loading
    # ------------------------------------------------------------------

    def load(self, *args, **kwargs):
        self._ensure_corpus()
        return super().load(*args, **kwargs)

    def record_to_sample(self, record: Dict[str, Any]) -> Sample:
        source_files = record.get('source_files', '')
        return Sample(
            input=record['question'],
            target=str(record['answer']),
            tools=[BASH_TOOL_INFO],
            metadata={
                'uid': record['uid'],
                'source_files': source_files,
            },
        )

    def _ensure_corpus(self) -> None:
        """Download the parsed txt corpus from ModelScope."""
        if self._corpus_dir is not None:
            return

        logger.info('Downloading OfficeQA corpus (parsed txt files)...')
        snapshot_dir = resolve_snapshot_or_local_path(
            self, allow_file_pattern=['treasury_bulletins_parsed/transformed/*.txt']
        )
        corpus_dir = os.path.join(snapshot_dir, 'treasury_bulletins_parsed', 'transformed')
        if not os.path.isdir(corpus_dir):
            raise FileNotFoundError(
                f'OfficeQA corpus not found at {corpus_dir}. '
                'Please ensure the dataset contains treasury_bulletins_parsed/transformed/*.txt files.'
            )
        self._corpus_dir = corpus_dir
        logger.info(f'OfficeQA corpus ready at: {self._corpus_dir}')

    # ------------------------------------------------------------------
    # Agent hooks
    # ------------------------------------------------------------------

    def build_tools(self, sample: Sample) -> Dict[str, Any]:
        return {'bash': run_bash}

    def build_environment(self, sample: Sample) -> Optional[AgentEnvironment]:
        if self._task_config.sandbox and self._task_config.sandbox.enabled:
            from evalscope.agent.environments.enclave import EnclaveAgentEnvironment
            sandbox_config = {
                'working_dir': _CONTAINER_CORPUS_DIR,
                'volumes': {
                    self._corpus_dir: {
                        'bind': _CONTAINER_CORPUS_DIR,
                        'mode': 'ro'
                    }
                },
            }
            return EnclaveAgentEnvironment(engine='docker', sandbox_config=sandbox_config)

        from evalscope.agent.environments.local import LocalAgentEnvironment
        return LocalAgentEnvironment(working_dir=self._corpus_dir)

    def build_initial_messages(self, sample: Sample) -> List[Any]:
        source_files = sample.metadata.get('source_files', '') if sample.metadata else ''
        input_text = INPUT_TEMPLATE.format(
            question=sample.input,
            source_files=source_files,
        )
        return [ChatMessageUser(content=input_text)]

    def build_strategy(self, sample: Any) -> Any:
        from evalscope.api.registry import get_strategy
        strategy_cls = get_strategy(self.strategy_name)
        corpus_dir = _CONTAINER_CORPUS_DIR if (
            self._task_config.sandbox and self._task_config.sandbox.enabled
        ) else self._corpus_dir
        return strategy_cls(system_prompt=SYSTEM_PROMPT.format(corpus_dir=corpus_dir))

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def extract_answer(self, prediction: str, task_state: TaskState) -> str:
        """Extract final answer from agent output."""
        return extract_final_answer(prediction)

    def match_score(
        self, original_prediction: str, filtered_prediction: str, reference: str, task_state: TaskState
    ) -> Score:
        """Score using official OfficeQA fuzzy numeric matching."""
        score = Score(extracted_prediction=filtered_prediction, prediction=original_prediction)
        correct = score_answer(reference, filtered_prediction, tolerance=DEFAULT_TOLERANCE)
        score.value = {'acc': correct}
        score.main_score_name = 'acc'
        return score
