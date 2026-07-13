# Copyright (c) Alibaba, Inc. and its affiliates.
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

from evalscope.api.benchmark import BenchmarkMeta, DefaultDataAdapter
from evalscope.api.dataset import DatasetDict, DatasetHub, Sample, build_dataset_from_records
from evalscope.api.evaluator import TaskState
from evalscope.api.messages import ChatMessageUser
from evalscope.api.metric import AggScore, SampleScore, Score
from evalscope.api.registry import register_benchmark
from evalscope.constants import Tags
from .utils import CATEGORY_IDS, CATEGORY_NAMES, DATA_FILE, build_qa_prompt, get_target_answer, locomo_f1_score


@register_benchmark(
    BenchmarkMeta(
        name='locomo',
        pretty_name='LoCoMo',
        tags=[Tags.QA, Tags.LONG_CONTEXT, Tags.MULTI_TURN],
        description="""
## Overview

LoCoMo evaluates very long-term conversational memory in two-person multi-session dialogues. This adapter supports the
official question-answering task from `locomo10.json`.

## Task Description

- **Task Type**: Long-context question answering
- **Input**: Multi-session conversation history with session dates and a question
- **Output**: Short free-form answer
- **Subsets**: `qa`

## Key Features

- Uses the official LoCoMo QA data file hosted on ModelScope
- Supports full-history long-context prompts and evidence-only oracle prompts
- Includes image captions from the released data when present, but does not download image files
- Uses LoCoMo's rule-based F1 / adversarial refusal scoring instead of an LLM judge

## Evaluation Notes

- Default subset is `qa` with `eval_mode=long_context`
- Use `extra_params.eval_mode='oracle_context'` for evidence-only upper-bound evaluation
- Event summarization, multimodal dialog generation, and RAG retrieval are not included in this QA adapter
""",
        dataset_id='evalscope/locomo',
        metric_list=['f1'],
        few_shot_num=0,
        train_split=None,
        eval_split='test',
        subset_list=['qa'],
        default_subset='qa',
        extra_params={
            'eval_mode': {
                'type': 'str',
                'description': 'Evaluation mode: long_context or oracle_context.',
                'value': 'long_context',
                'choices': ['long_context', 'oracle_context'],
            },
        },
    )
)
class LoCoMoAdapter(DefaultDataAdapter):
    """Adapter for the LoCoMo QA task."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.eval_mode = self.extra_params.get('eval_mode', 'long_context')

    def load(self) -> Tuple[DatasetDict, None]:
        self._validate_params()
        records = self._load_records()
        dataset = build_dataset_from_records(
            records=records,
            sample_fields=self.record_to_sample,
            name='qa',
            location=self.dataset_id,
            limit=self.limit,
            repeats=self.repeats,
            shuffle=self.shuffle,
            seed=self.seed,
        )
        return DatasetDict({'qa': dataset}), None

    def _validate_params(self) -> None:
        if self.subset_list != ['qa']:
            raise ValueError('LoCoMo currently supports subset_list=["qa"] only.')
        if self.eval_mode not in ['long_context', 'oracle_context']:
            raise ValueError(f'Unsupported eval_mode: {self.eval_mode}')

    def _load_records(self) -> List[Dict[str, Any]]:
        file_path = self._resolve_dataset_file()
        conversations = json.loads(Path(file_path).read_text(encoding='utf-8'))
        records = []
        for conversation_record in conversations:
            conversation = conversation_record['conversation']
            sample_id = conversation_record['sample_id']
            for qa_index, qa in enumerate(conversation_record['qa']):
                records.append({
                    'sample_id': sample_id,
                    'qa_index': qa_index,
                    'conversation': conversation,
                    'qa': qa,
                })
        return records

    def _resolve_dataset_file(self) -> str:
        dataset_path = Path(self.dataset_id)
        if dataset_path.is_file():
            return str(dataset_path)
        if dataset_path.exists():
            file_path = dataset_path / DATA_FILE
            if not file_path.exists():
                raise FileNotFoundError(f'LoCoMo data file not found: {file_path}')
            return str(file_path)
        return DatasetHub(
            data_id_or_path=self.dataset_id,
            data_source=self.dataset_hub,
            force_redownload=self.force_redownload,
            cache_dir=self.dataset_dir,
        ).download_file(DATA_FILE)

    def record_to_sample(self, record: Dict[str, Any]) -> Sample:
        qa = record['qa']
        prompt, prompt_metadata = build_qa_prompt(
            conversation=record['conversation'],
            qa=qa,
            eval_mode=self.eval_mode,
        )
        target = get_target_answer(qa)
        metadata = {
            'sample_id': record['sample_id'],
            'qa_index': record['qa_index'],
            'category': qa['category'],
            'category_name': CATEGORY_NAMES.get(qa['category'], f'category_{qa["category"]}'),
            'raw_question': qa['question'],
            'answer': target,
            'adversarial_answer': qa.get('adversarial_answer'),
            'eval_mode': self.eval_mode,
            **prompt_metadata,
        }
        return Sample(input=[ChatMessageUser(content=prompt)], target=target, metadata=metadata)

    def match_score(
        self, original_prediction: str, filtered_prediction: str, reference: str, task_state: TaskState
    ) -> Score:
        category = int(task_state.metadata['category'])
        value = locomo_f1_score(filtered_prediction, reference, category)
        score = Score(extracted_prediction=filtered_prediction, prediction=original_prediction)
        score.value = {'f1': value}
        score.metadata = {
            'category': category,
            'category_name': task_state.metadata.get('category_name'),
        }
        score.main_score_name = 'f1'
        return score

    def aggregate_scores(self, sample_scores: List[SampleScore]) -> List[AggScore]:
        valid_scores = [s for s in sample_scores if s.score and 'f1' in s.score.value]
        if not valid_scores:
            return []

        agg_scores = [
            self._make_agg_score(metric_name='f1', aggregation_name='overall', scores=valid_scores),
        ]

        category_means = []
        for category in CATEGORY_IDS:
            category_scores = [s for s in valid_scores if (s.sample_metadata or {}).get('category') == category]
            if not category_scores:
                continue
            category_name = CATEGORY_NAMES.get(category, f'category_{category}')
            agg = self._make_agg_score(metric_name='f1', aggregation_name=category_name, scores=category_scores)
            agg_scores.append(agg)
            category_means.append(agg.score)

        if category_means:
            agg_scores.append(
                AggScore(
                    metric_name='f1',
                    aggregation_name='task_averaged',
                    score=sum(category_means) / len(category_means),
                    num=len(category_means),
                )
            )

        return agg_scores

    @staticmethod
    def _make_agg_score(metric_name: str, aggregation_name: str, scores: List[SampleScore]) -> AggScore:
        values = [float(s.score.value[metric_name]) for s in scores]
        ids = [s.sample_id for s in scores]
        return AggScore(
            metric_name=metric_name,
            aggregation_name=aggregation_name,
            score=sum(values) / len(values),
            num=len(values),
            ids=ids,
        )
