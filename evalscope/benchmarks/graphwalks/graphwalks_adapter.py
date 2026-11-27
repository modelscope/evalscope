import json
import re
from typing import Any

from evalscope.api.benchmark import BenchmarkMeta, DefaultDataAdapter
from evalscope.api.dataset import Sample
from evalscope.api.evaluator import TaskState
from evalscope.api.metric import Score
from evalscope.api.registry import register_benchmark
from evalscope.constants import Tags

DATASET_REVISION = 'be6cc6ecf9b4d495b07d1ff53d2a16598e90fed7'


@register_benchmark(
    BenchmarkMeta(
        name='graphwalks',
        pretty_name='GraphWalks',
        evaluation_version='v1.0',
        dataset_id='openai/graphwalks',
        hf_dataset_id='openai/graphwalks',
        dataset_revision=DATASET_REVISION,
        tags=[Tags.REASONING, Tags.LONG_CONTEXT],
        subset_list=['bfs', 'parents'],
        eval_split='train',
        few_shot_num=0,
        few_shot_mode='disabled',
        prompt_template='{question}',
        metric_list=['f1', 'precision', 'recall'],
        primary_metric='f1',
        aggregation='mean',
        description="""
## Overview

GraphWalks is OpenAI's long-context graph reasoning benchmark. Models receive a directed graph's edge list and return the nodes resulting from a graph operation.

## Task Description

- **Task Type**: Graph traversal and parent-node retrieval
- **Input**: The official user prompt, including three demonstrations, a directed edge list, and an operation
- **Output**: A final line containing `Final Answer: [node1, node2, ...]`
- **Domain**: Long-context reasoning over graphs

## Key Features

- All 1,150 examples from both official Parquet files: `graphwalks_128k_and_shorter.parquet` and `graphwalks_256k_to_1mil.parquet`.
- Separate `bfs` (breadth-first search) and `parents` result subsets.
- MIT-licensed data pinned to revision `be6cc6ecf9b4d495b07d1ff53d2a16598e90fed7`, including the February 2026 parent-answer corrections and clarified BFS prompts.
- Prompts pass through unchanged. `few_shot_num=0` means no additional demonstrations; the published prompts already contain three.

## Evaluation Notes

- Use `--dataset-hub huggingface` to load the official dataset. The evaluation split is `train`; both files are loaded by the standard dataset loader. No ModelScope mirror is declared.
- Primary metric: mean per-example **F1**, with mean per-example **precision** and **recall**. Overall results weight task subsets by their sample counts, rather than pooling node counts.
- Extraction and scoring follow the [pinned OpenAI reference](https://huggingface.co/datasets/openai/graphwalks/blob/be6cc6ecf9b4d495b07d1ff53d2a16598e90fed7/README.md). Node order and duplicates do not affect scores.
- Only the literal last line is parsed, using a case-sensitive `Final Answer:` substring and a greedy bracket match. Trailing newlines fail parsing; node quotes are not removed. No response cleanup is enabled by default.
- A successfully parsed empty list against an empty target scores 1 on all metrics. Parse failures score 0, including against an empty target, and remain in the denominator.
- Real-model runs require long-context inference: the files cover nominal contexts through 1M tokens, and prompts can exceed 1.7 million characters. Character counts are not token counts. Check the model's tokenizer, chat-template overhead, input limit, and output-token headroom before running. The adapter does not truncate prompts; configure the backend to reject over-context requests and report failed or omitted samples when comparing results.
- No GPU, judge, or paid API is needed for the tests. Offline scoring and mocked end-to-end checks: `pytest tests/benchmark/test_graphwalks.py`. Full official-data verification (downloads about 319 MB on first use): `EVALSCOPE_GRAPHWALKS_DOWNLOAD=1 pytest tests/benchmark/test_graphwalks.py -k official_dataset`.
- A small live-data pipeline smoke run is `evalscope eval --datasets graphwalks --dataset-hub huggingface --model dummy --eval-type mock_llm --limit 1`. The limit applies per task subset; mock scores do not measure model quality.
""",
    )
)
class GraphWalksAdapter(DefaultDataAdapter):
    """Load official graph operations and reproduce OpenAI's node-set grader."""

    doc_dataset_hub = 'huggingface'

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.reformat_subset = True

    def record_to_sample(self, record: dict[str, Any]) -> Sample:
        """Preserve the published prompt and keep the node set as one structured target."""
        return Sample(
            input=record['prompt'],
            target=json.dumps(record['answer_nodes']),
            subset_key=record['problem_type'],
            metadata={'problem_type': record['problem_type'], 'prompt_chars': record['prompt_chars']},
        )

    def extract_answer(self, prediction: str, task_state: TaskState) -> str:
        """Return a JSON node list, or an empty string for a reference-parser failure."""
        # Keep the reference's literal last line and greedy regex, including their edge cases.
        line = prediction.split('\n')[-1]
        if 'Final Answer:' not in line:
            return ''
        list_part = re.search(r'\[.*\]', line)
        if list_part is None:
            return ''
        nodes = list_part.group(0).strip('[]').split(',')
        return json.dumps([node.strip() for node in nodes if node.strip()])

    def match_score(
        self, original_prediction: str, filtered_prediction: str, reference: str, task_state: TaskState
    ) -> Score:
        """Score node sets, distinguishing a parsed empty list from a parse failure."""
        failed_to_parse = filtered_prediction == ''
        sampled_set = set(json.loads(filtered_prediction)) if not failed_to_parse else set()
        truth_set = set(json.loads(reference))
        precision = recall = f1 = 0.0
        if not failed_to_parse:
            if not truth_set and not sampled_set:
                precision = recall = f1 = 1.0
            else:
                overlap = len(sampled_set & truth_set)
                recall = overlap / len(truth_set) if truth_set else 0.0
                precision = overlap / len(sampled_set) if sampled_set else 0.0
                f1 = 2 * (recall * precision) / (recall + precision) if recall + precision > 0 else 0.0
        return Score(
            value={'f1': f1, 'precision': precision, 'recall': recall},
            main_score_name='f1',
            prediction=original_prediction,
            extracted_prediction=filtered_prediction,
            metadata={'failed_to_parse': failed_to_parse},
        )
