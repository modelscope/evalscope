# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import Any, Dict, List

from evalscope.api.benchmark import BenchmarkMeta, DefaultDataAdapter
from evalscope.api.dataset import Sample
from evalscope.api.evaluator import TaskState
from evalscope.api.metric import AggScore, SampleScore, Score
from evalscope.api.registry import register_benchmark
from evalscope.constants import ScoreStatus, Tags
from evalscope.report import Report

from .scoring import score_response

TASKS = ('1-1', '1-2', *(f'2-{i}' for i in range(2, 11)), *(f'3-{i}' for i in range(1, 9)))
CATEGORIES = {'1': 'memorization', '2': 'understanding', '3': 'application'}


@register_benchmark(
    BenchmarkMeta(
        name='lawbench',
        pretty_name='LawBench',
        dataset_id='evalscope/LawBench',
        paper_url='https://arxiv.org/abs/2309.16289',
        tags=[Tags.KNOWLEDGE, Tags.REASONING, Tags.QA],
        subset_list=list(TASKS),
        eval_split='zero_shot',
        prompt_template='{question}',
        few_shot_mode='disabled',
        metric_list=['lawbench_score', 'abstention_rate'],
        primary_metric='lawbench_score',
        evaluation_version='v1.0',
        description="""
## Overview

LawBench evaluates Chinese legal knowledge and reasoning across memorization, understanding and application. This integration supports 19 tasks, including legal exams, case analysis, information extraction and legal consultation. See the [official project](https://github.com/open-compass/LawBench) for the benchmark and paper.

## Task Description

- **Task Type**: Legal question answering, classification, extraction and generation
- **Input**: A Chinese legal question or document with task instructions
- **Output**: A Chinese response in the requested format
- **Domain**: Chinese legal knowledge and legal reasoning

## Key Features

- 9,500 supported samples, with 500 per task; select tasks using subsets `1-1`–`1-2`, `2-2`–`2-10`, and `3-1`–`3-8`
- Task groups cover legal knowledge, document understanding and practical application
- Data downloads automatically from [ModelScope](https://modelscope.cn/datasets/evalscope/LawBench)
- Uses the official task-specific scoring rules; no LLM judge is required

## Evaluation Notes

- Zero-shot evaluation of 19 tasks; document proofreading (`2-1`) is not supported.
- Install dependencies with `pip install 'evalscope[lawbench]'`. For offline use, set `local_path` to a ModelScope dataset snapshot.
- `lawbench_score`: higher is better (usually 0–1; imprisonment scores may be negative). Overall is the unweighted mean across evaluated tasks.
- `abstention_rate`: share of responses with no valid extracted answer. Officially excluded references are not scored.
""",
    )
)
class LawBenchAdapter(DefaultDataAdapter):
    """Native adapter for the official LawBench zero-shot task collection."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        if '2-1' in self.subset_list:
            raise ValueError('LawBench task 2-1 (document proofreading) is not supported. Select other subsets.')
        unknown = set(self.subset_list) - set(TASKS)
        if unknown:
            raise ValueError(f'Unknown LawBench subsets: {sorted(unknown)}')
        if self.eval_split != 'zero_shot':
            raise ValueError('LawBench supports only the zero_shot split.')
        if self.aggregation != 'mean':
            raise ValueError('LawBench requires mean aggregation for its official task scores.')
        self.category_map = {task: [CATEGORIES[task[0]]] for task in TASKS}

    def record_to_sample(self, record: Dict[str, Any]) -> Sample:
        for field in ('instruction', 'question', 'answer'):
            if not isinstance(record.get(field), str):
                raise ValueError(f'LawBench record requires a string {field!r}.')
        task = record.get('task', self.current_subset_name)
        if not isinstance(task, str) or task not in TASKS:
            raise ValueError(f'Unknown LawBench task: {task!r}')
        prompt = f'{record["instruction"]}\n{record["question"]}'
        return Sample(input=prompt, target=record['answer'], metadata={'task': task})

    def match_score(
        self, original_prediction: str, filtered_prediction: str, reference: str, task_state: TaskState
    ) -> Score:
        task = task_state.metadata['task']
        values = score_response(task, filtered_prediction, reference)
        if 'score' in values:
            values['lawbench_score'] = values.pop('score')
        return Score(
            value=values,
            main_score_name='lawbench_score',
            prediction=original_prediction,
            extracted_prediction=filtered_prediction,
        )

    def aggregate_scores(self, sample_scores: List[SampleScore]) -> List[AggScore]:
        scores = [s for s in sample_scores if s.score.status == ScoreStatus.SUCCESS]
        if not scores:
            return []
        tasks = {s.sample_metadata['task'] for s in scores}
        if len(tasks) != 1:
            raise ValueError('LawBench aggregation requires scores from exactly one task.')
        return super().aggregate_scores(scores)

    def _on_generate_report(self, scores: Dict[str, List[AggScore]], model_name: str) -> Report:
        report = super()._on_generate_report(scores, model_name)
        for metric in report.metrics:
            values = [
                agg.score for aggregates in scores.values() for agg in aggregates if agg.identity == metric.identity
            ]
            metric.score = sum(values) / len(values)
        return report
