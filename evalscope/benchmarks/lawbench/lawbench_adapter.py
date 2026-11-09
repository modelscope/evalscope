# Copyright (c) Alibaba, Inc. and its affiliates.
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Type

from evalscope.api.benchmark import BenchmarkMeta, DefaultDataAdapter
from evalscope.api.dataset import DataLoader, Dataset, DictDataLoader, Sample, download_dataset_file
from evalscope.api.evaluator import TaskState
from evalscope.api.metric import AggScore, SampleScore, Score
from evalscope.api.model import Model
from evalscope.api.registry import register_benchmark
from evalscope.constants import HubType, ScoreStatus, Tags
from evalscope.report import Report

from .cherrant import score_corrections, validate_toolkit
from .scoring import score_response

TASKS = ('1-1', '1-2', *(f'2-{i}' for i in range(1, 11)), *(f'3-{i}' for i in range(1, 9)))
CATEGORIES = {'1': 'memorization', '2': 'understanding', '3': 'application'}


@register_benchmark(
    BenchmarkMeta(
        name='lawbench',
        pretty_name='LawBench',
        dataset_id='LawBench',
        paper_url='https://arxiv.org/abs/2309.16289',
        tags=[Tags.KNOWLEDGE, Tags.REASONING, Tags.QA],
        subset_list=list(TASKS),
        eval_split='zero_shot',
        prompt_template='{question}',
        few_shot_mode='disabled',
        metric_list=['lawbench_score', 'abstention_rate'],
        primary_metric='lawbench_score',
        evaluation_version='v1.0',
        extra_params={
            'cherrant_path': {
                'type': 'str',
                'value': '',
                'description': 'Official LawBench evaluation/utils directory; required for task 2-1.',
            },
            'cherrant_python': {
                'type': 'str',
                'value': '',
                'description': 'Python executable with ChERRANT dependencies; defaults to the current interpreter.',
            },
            'cherrant_timeout': {
                'type': 'float',
                'value': 600.0,
                'description': 'Timeout in seconds for each ChERRANT subprocess.',
            },
        },
        description="""
## Overview

LawBench evaluates Chinese legal knowledge through 20 tasks covering memorization, understanding and application. This native adapter follows the [official zero-shot data and scoring code](https://github.com/open-compass/LawBench) at revision `e30981bb3ff54c41571f222e0b23e92d27375388`.

## Task Description

- **Task Type**: Legal question answering, classification, extraction, generation and proofreading
- **Input**: The official instruction followed by the question, separated by a newline
- **Output**: A free-form Chinese response in the task's requested format
- **Domain**: Chinese legal knowledge and legal reasoning

## Key Features

- 10,000 samples: 500 per task; subsets `1-1`–`1-2`, `2-1`–`2-10`, and `3-1`–`3-8`
- Covers article recitation, legal exams, proofreading, dispute classification, reading comprehension, entity/event extraction, summarization, argument mining, judgment prediction, damages calculation and consultation
- Preserves official substring-based answer parsing, character F1, ROUGE-L and normalized imprisonment log distance
- Task `2-1` uses official ChERRANT character-level corpus F0.5, computed once over the evaluated records

## Evaluation Notes

- Install the ordinary scoring dependency with `pip install 'evalscope[lawbench]'`.
- Download the official repository and set `dataset_args.lawbench.local_path` to its root, `data/zero_shot` directory, or another directory containing the 20 `<task>.json` files. There is no automatic complete hub mirror in this adapter.
- Zero-shot only. Select individual tasks with `subset_list`; standard `limit`, `shuffle`, filters and `repeats` apply.
- `lawbench_score` is the task's official score, usually in [0, 1]; imprisonment log-distance scores can be negative. Overall and category scores are unweighted means of represented task scores.
- Officially skipped dispute/imprisonment references are omitted from score denominators but retained in abstention denominators. NER's upstream `anstention_rate` spelling is normalized to `abstention_rate`; the official CSV driver instead silently defaults this misspelled field to zero.
- Task `2-1` additionally requires the official `evaluation/utils` directory and its ChERRANT dependencies (`ltp`, `pypinyin`, `opencc-python-reimplemented`, `timeout-decorator`, `python-Levenshtein`). Set `extra_params.cherrant_path` to that directory and optionally `cherrant_python` to a separate Python interpreter. Scoring runs in private temporary directories; tool errors and timeouts fail explicitly.
- No LLM judge or OpenCompass runtime is required. This is LawBench, distinct from PLawBench.
- The generic Usage examples below must also include `local_path` and, when selecting task `2-1`, `cherrant_path`. The complete configuration is:

```python
from evalscope import TaskConfig, run_task

run_task(TaskConfig(
    model='your-model', eval_type='openai_api',
    api_url='http://localhost:8000/v1', api_key='EMPTY',
    datasets=['lawbench'],
    dataset_args={'lawbench': {
        'local_path': '/path/to/LawBench',
        'extra_params': {'cherrant_path': '/path/to/LawBench/evaluation/utils'},
    }},
    generation_config={'temperature': 0, 'max_tokens': 1024},
))
```
""",
    )
)
class LawBenchAdapter(DefaultDataAdapter):
    """Native adapter for the official LawBench zero-shot task collection."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        unknown = set(self.subset_list) - set(TASKS)
        if unknown:
            raise ValueError(f'Unknown LawBench subsets: {sorted(unknown)}')
        if self.eval_split != 'zero_shot':
            raise ValueError('LawBench supports only the zero_shot split.')
        if self.aggregation != 'mean':
            raise ValueError('LawBench requires mean aggregation for its official task scores.')
        self.category_map = {task: [CATEGORIES[task[0]]] for task in TASKS}

    def load_subset(self, subset: str, data_loader: Type[DataLoader]) -> Dataset:
        """Read official JSON and delegate sampling, filtering and indexing to the standard loader."""
        if subset not in TASKS:
            raise ValueError(f'Unknown LawBench subset: {subset!r}')
        root = Path(self.dataset_id).expanduser()
        if not root.is_dir():
            raise FileNotFoundError(
                'LawBench requires local_path pointing to the official checkout or zero_shot data directory.'
            )
        for prefix in ('data/zero_shot', 'zero_shot', ''):
            if (root / prefix / f'{subset}.json').is_file():
                break
        path = download_dataset_file(str(root), f'{prefix}/{subset}.json'.lstrip('/'), data_source=HubType.LOCAL)
        records = json.loads(Path(path).read_text(encoding='utf-8'))
        if not isinstance(records, list):
            raise ValueError('LawBench task JSON must contain a list of records.')
        return DictDataLoader(
            dict_list=[{**record, 'task': subset} for record in records],
            sample_fields=self.record_to_sample,
            filter_func=self.sample_filter,
            limit=self.limit,
            repeats=self.repeats,
            shuffle=self.shuffle,
            seed=self.seed,
        ).load()

    def record_to_sample(self, record: Dict[str, Any]) -> Sample:
        for field in ('instruction', 'question', 'answer', 'task'):
            if not isinstance(record.get(field), str):
                raise ValueError(f'LawBench record requires a string {field!r}.')
        prompt = f'{record["instruction"]}\n{record["question"]}'
        metadata = {'task': record['task']}
        if record['task'] == '2-1':
            metadata.update(origin_prompt=prompt, reference=record['answer'])
        return Sample(input=prompt, target=record['answer'], metadata=metadata)

    def _on_inference_start(self, model: Model, sample: Sample) -> None:
        if sample.metadata['task'] == '2-1':
            validate_toolkit(Path(self.extra_params['cherrant_path']))
        super()._on_inference_start(model, sample)

    def match_score(
        self, original_prediction: str, filtered_prediction: str, reference: str, task_state: TaskState
    ) -> Score:
        task = task_state.metadata['task']
        if task == '2-1':
            return Score(
                prediction=original_prediction,
                extracted_prediction=filtered_prediction,
                metadata={'corpus_metric': 'cherrant_f0.5'},
            )
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
        if tasks == {'2-1'}:
            rows = [
                dict(
                    origin_prompt=s.sample_metadata['origin_prompt'],
                    prediction=s.score.extracted_prediction
                    if s.score.extracted_prediction is not None
                    else s.score.prediction,
                    refr=s.sample_metadata['reference'],
                )
                for s in scores
            ]
            value = score_corrections(
                rows,
                Path(self.extra_params['cherrant_path']),
                self.extra_params['cherrant_python'] or sys.executable,
                float(self.extra_params['cherrant_timeout']),
            )
            return [
                AggScore(
                    metric_name='lawbench_score',
                    aggregation='mean',
                    score=value,
                    num=len(scores),
                    metadata={'official_metric': 'corpus_f0.5'},
                )
            ]
        return super().aggregate_scores(scores)

    def _on_generate_report(self, scores: Dict[str, List[AggScore]], model_name: str) -> Report:
        report = super()._on_generate_report(scores, model_name)
        for metric in report.metrics:
            values = [
                agg.score for aggregates in scores.values() for agg in aggregates if agg.identity == metric.identity
            ]
            metric.score = sum(values) / len(values)
            for category in metric.categories:
                category.score = category.macro_score
        return report
