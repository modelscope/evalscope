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
                'description': 'Maximum wait in seconds for each proofreading scoring step.',
            },
        },
        description="""
## Overview

LawBench evaluates Chinese legal knowledge and reasoning through 20 tasks covering memorization, understanding and application. It includes legal exams, case analysis, information extraction, document proofreading and legal consultation. See the [official project](https://github.com/open-compass/LawBench) for the benchmark and paper.

## Task Description

- **Task Type**: Legal question answering, classification, extraction, generation and proofreading
- **Input**: A Chinese legal question or document with task instructions
- **Output**: A Chinese response in the requested format
- **Domain**: Chinese legal knowledge and legal reasoning

## Key Features

- 10,000 samples, with 500 per task; select tasks using subsets `1-1`–`1-2`, `2-1`–`2-10`, and `3-1`–`3-8`
- Task groups cover legal knowledge, document understanding and practical application
- Data downloads automatically from [ModelScope](https://modelscope.cn/datasets/evalscope/LawBench)
- Uses the official task-specific scoring rules; no LLM judge is required

## Evaluation Notes

- Supports zero-shot evaluation. Set `subset_list` to evaluate selected tasks; use `limit` for a short trial.
- `lawbench_score` follows each task's official metric. Higher is better; most scores are in [0, 1], while imprisonment prediction scores can be negative. Overall and category scores give equal weight to each evaluated task.
- `abstention_rate` reports how often a task's answer parser finds no valid answer. Officially excluded reference answers do not contribute to the score.
- This benchmark is distinct from PLawBench and does not require OpenCompass.
- For offline data, set `local_path` to the official checkout, its `data/zero_shot` directory, or a directory containing the original `<task>.json` files.

### Install and run ordinary tasks

Install the ordinary scoring dependencies:

```bash
python -m pip install 'evalscope[lawbench]'
```

From an EvalScope source checkout, use `python -m pip install -e '.[lawbench]'` instead.
The following example evaluates the legal exam subset; it does not need ChERRANT:

```python
from evalscope import TaskConfig, run_task

run_task(TaskConfig(
    model='your-model', eval_type='openai_api',
    api_url='http://localhost:8000/v1', api_key='EMPTY',
    datasets=['lawbench'],
    dataset_args={'lawbench': {'subset_list': ['1-2']}},
    limit=2,
    generation_config={'temperature': 0, 'max_tokens': 1024},
))
```

### Set up document proofreading (task 2-1)

Task `2-1` uses ChERRANT to compare text corrections and compute F0.5 over the evaluated records. It needs additional Python packages and the official tool files; CPU execution is sufficient. The other 19 tasks do not need this setup.

Install the optional dependencies and download the compatible tool version:

```bash
python -m pip install 'evalscope[lawbench-cherrant]'
git clone --filter=blob:none --sparse https://github.com/open-compass/LawBench.git
git -C LawBench sparse-checkout set evaluation/utils
git -C LawBench checkout e30981bb3ff54c41571f222e0b23e92d27375388
```

From an EvalScope source checkout, replace the first command with `python -m pip install -e '.[lawbench-cherrant]'`.
Run the following example from the directory containing the downloaded `LawBench` folder. To evaluate all 20 tasks, remove `subset_list`.

```python
from pathlib import Path

from evalscope import TaskConfig, run_task

run_task(TaskConfig(
    model='your-model', eval_type='openai_api',
    api_url='http://localhost:8000/v1', api_key='EMPTY',
    datasets=['lawbench'],
    dataset_args={'lawbench': {
        'subset_list': ['2-1'],
        'extra_params': {
            'cherrant_path': str(Path('LawBench/evaluation/utils').resolve()),
            # 'cherrant_python': '/absolute/path/to/cherrant-env/bin/python',
        },
    }},
    limit=2,
    generation_config={'temperature': 0, 'max_tokens': 1024},
))
```

If these dependencies are installed in a separate Python environment, set `extra_params.cherrant_python` to that environment's absolute Python executable path. Install the optional dependencies with that interpreter's `python -m pip`; creating a separate environment is optional.
The generic Usage examples below select all tasks, so they also need `cherrant_path` configured as shown above.
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
        """Use the standard hub loader, with support for original local JSON directories."""
        if subset not in TASKS:
            raise ValueError(f'Unknown LawBench subset: {subset!r}')
        root = Path(self.dataset_id).expanduser()
        if not root.is_dir():
            return super().load_subset(subset, data_loader)
        for prefix in ('data/zero_shot', 'zero_shot', ''):
            if (root / prefix / f'{subset}.json').is_file():
                break
        path = download_dataset_file(
            str(root), f'{prefix}/{subset}.json'.lstrip('/'), data_source=HubType.LOCAL, revision=self.dataset_revision
        )
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
        for field in ('instruction', 'question', 'answer'):
            if not isinstance(record.get(field), str):
                raise ValueError(f'LawBench record requires a string {field!r}.')
        task = record.get('task', self.current_subset_name)
        if not isinstance(task, str) or task not in TASKS:
            raise ValueError(f'Unknown LawBench task: {task!r}')
        prompt = f'{record["instruction"]}\n{record["question"]}'
        metadata = {'task': task}
        if task == '2-1':
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
