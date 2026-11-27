# Copyright (c) Alibaba, Inc. and its affiliates.
"""Dense Prompt Graph Benchmark, introduced in ELLA (arXiv:2403.05135)."""

import csv
import os
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Literal, Sequence, Type

from pydantic import BaseModel

from evalscope.api.benchmark import BenchmarkMeta, Text2ImageAdapter
from evalscope.api.dataset import DataLoader, Dataset, DictDataLoader, Sample
from evalscope.api.judge import (
    CaseVerdict,
    JudgeCase,
    JudgeContext,
    JudgeDefinition,
    JudgeRequest,
    JudgeReview,
    OutputContract,
    Placement,
    ReducedVerdict,
)
from evalscope.api.messages import ChatMessageUser, ContentImage, ContentText
from evalscope.api.metric import MetricSelector, Score
from evalscope.api.registry import register_benchmark
from evalscope.benchmarks.text2image.corpus_aggregation import CorpusAggregationMixin
from evalscope.constants import FileConstants, ScoreStatus, ScoringPolicy, Tags
from evalscope.metrics.vision import dpg_scoring

_METRIC = 'dpg_score'


class _DPGVerdict(BaseModel):
    answer: Literal['yes', 'no']


_DPG_CONTRACT = OutputContract(schema_model=_DPGVerdict, allow_json_fences=True)


def convert_dpg_csv(csv_path: str) -> List[Dict[str, Any]]:
    """Group the official ``dpg_bench.csv`` rows into prompt-level records.

    Each record carries the prompt and its atomic question metadata (proposition id,
    question text, dependency ids, L1/L2 categories), following the official
    question_dict structure built by the reference evaluator.
    """
    by_item: Dict[str, Dict[str, Any]] = {}
    prompt_files = {path.stem: path for path in (Path(csv_path).parent / 'prompts').glob('*.txt')}
    with open(csv_path, encoding='utf-8', newline='') as handle:
        for row in csv.DictReader(handle):
            item_id = row['item_id']
            if item_id not in by_item:
                prompt_file = prompt_files.get(item_id)
                prompt = prompt_file.read_bytes().decode('utf-8') if prompt_file is not None else row['text']
                by_item[item_id] = {'item_id': item_id, 'prompt': prompt, 'questions': []}
            by_item[item_id]['questions'].append(
                {
                    'proposition_id': int(row['proposition_id']),
                    'question': row['question_natural_language'],
                    'dependency': [int(parent) for parent in row['dependency'].split(',') if parent.strip()],
                    'category_broad': row['category_broad'],
                    'category_detailed': row['category_detailed'],
                }
            )
    return list(by_item.values())


@register_benchmark(
    BenchmarkMeta(
        name='dpg_bench',
        pretty_name='DPG-Bench',
        paper_url='https://arxiv.org/abs/2403.05135',
        dataset_id='evalscope/DPG-Bench',
        dataset_revision='a432ab3c902d8d12ba4ef14371c9e933b75724dd',
        description="""
## Overview

DPG-Bench evaluates how well generated images follow dense prompts about objects, attributes, and relations.

## Task Description

- **Task Type**: Text-to-image generation
- **Input**: Dense prompt describing a multi-object scene
- **Output**: Image scored through atomic yes/no questions decomposed from the prompt
- **Domain**: Dense prompt semantic alignment

## Key Features

- 1,065 official prompts and 14,392 yes/no questions with dependencies.
- The ModelScope mirror includes the exact prompts. For a local CSV, place the original text files in `prompts/`.

## Evaluation Notes

- Configure a vision judge with `--judge`. Missing or invalid verdicts exclude the image. The official judge is mPLUG.
- Overall score averages dependency-adjusted image scores. Category scores pool raw verdicts across images.
- EvalScope retains all CSV rows and uses every image for category scoring. The reference script skips its first CSV
  row and uses the final crop for category scoring. Use `repeats=4` to generate four images per prompt.
""",
        tags=[Tags.TEXT_TO_IMAGE],
        subset_list=['default'],
        metric_list=[_METRIC],
        primary_metric=MetricSelector(name=_METRIC, aggregation='mean', dimensions={'scope': 'overall'}),
        few_shot_num=0,
        train_split=None,
        eval_split='test',
        evaluation_version='v1.0',
    )
)
class DPGBenchAdapter(CorpusAggregationMixin, Text2ImageAdapter):
    scoring_policy = ScoringPolicy.JUDGE_ONLY
    corpus_metrics = {_METRIC: 'components'}

    def load_subset(self, subset: str, data_loader: Type[DataLoader]) -> Dataset:
        """Convert a local CSV in memory and use the standard loader for prepared datasets."""
        if not os.path.isfile(self.dataset_id) or not self.dataset_id.lower().endswith('.csv'):
            return super().load_subset(subset, data_loader)
        return DictDataLoader(
            dict_list=convert_dpg_csv(self.dataset_id),
            subset=subset,
            sample_fields=self.record_to_sample,
            limit=self.limit,
            repeats=self.repeats,
            shuffle=self.shuffle,
            seed=self.seed,
            filter_func=self.sample_filter,
        ).load()

    def record_to_sample(self, record: Dict[str, Any]) -> Sample:
        """Carry the official item identifier and atomic questions in sample metadata."""
        return super().record_to_sample(
            {'id': record['item_id'], 'prompt': record['prompt'], 'questions': record['questions']}
        )

    def judge_definition(self, context: JudgeContext) -> JudgeDefinition:
        """Judge all atomic questions before applying dependency and category scoring."""
        metadata = context.task_state.metadata or {}
        image_path = metadata.get(FileConstants.IMAGE_PATH, '')
        questions = dpg_scoring.parse_questions(metadata.get('questions', []))
        if not questions or not Path(image_path).is_file():
            reason = 'The sample must have atomic questions and a generated image.'
            score = Score(prediction=image_path, status=ScoreStatus.EXCLUDED, explanation=reason)
            return JudgeDefinition.skip(score, reason=reason)

        cases = [
            JudgeCase(case_id=str(question['qid']), output_contract=_DPG_CONTRACT, metadata=question)
            for question in questions
        ]
        category_counts = Counter(question['l1'] for question in questions)

        def request(
            case: JudgeCase,
            placement: Placement,
            completed: Sequence[CaseVerdict],
            judge_context: JudgeContext,
        ) -> JudgeRequest:
            prompt = case.metadata['question'] + case.output_contract.instruction()
            return JudgeRequest(
                messages=[ChatMessageUser(content=[ContentImage(image=image_path), ContentText(text=prompt)])]
            )

        def reduce(verdicts: Sequence[CaseVerdict], judge_context: JudgeContext) -> ReducedVerdict:
            answers = {verdict.metadata['qid']: verdict.value.answer for verdict in verdicts}
            raw, adjusted = dpg_scoring.raw_and_adjusted(questions, answers)
            overall, _, _ = dpg_scoring.overall_counts(questions, adjusted)
            values = {_METRIC: overall}
            for category in dpg_scoring.L1_CATEGORIES:
                if category_counts[category]:
                    category_score, _, _ = dpg_scoring.category_raw_counts(questions, raw, category)
                    values[f'{_METRIC}:{category}'] = category_score
            return ReducedVerdict(value=values)

        def finalize(score: Score, review: JudgeReview, judge_context: JudgeContext) -> Score:
            score.prediction = score.extracted_prediction = image_path
            if not score.value:
                return score
            values = {key.partition(':')[2] or 'overall': value for key, value in score.value.items()}
            components = {
                category: {'numerator': values[category] * count, 'denominator': count}
                for category, count in category_counts.items()
                if category in values
            }
            score.value.clear()
            metric_score = Score(value=values, metadata={'corpus_counts': {'components': components}})
            self._record_metric_result(score, _METRIC, metric_score)
            return score

        return JudgeDefinition.workflow(
            cases=cases,
            request=request,
            reduce=reduce,
            main_score_name=_METRIC,
            finalize=finalize,
        )
