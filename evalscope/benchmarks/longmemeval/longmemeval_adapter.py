# Copyright (c) Alibaba, Inc. and its affiliates.
import json
from pathlib import Path
from pydantic import BaseModel
from typing import Any, Dict, List, Literal, Tuple, Union

from evalscope.api.benchmark import BenchmarkMeta, DefaultDataAdapter
from evalscope.api.dataset import DatasetDict, DatasetHub, MemoryDataset, Sample, build_dataset_from_records
from evalscope.api.evaluator import TaskState
from evalscope.api.judge import (
    CaseVerdict,
    JudgeCase,
    JudgeContext,
    JudgeDefinition,
    JudgeRequest,
    OutputContract,
    ReducedVerdict,
)
from evalscope.api.messages import ChatMessageUser
from evalscope.api.metric import AggScore, SampleScore, Score
from evalscope.api.metric.semantics import MetricSelector
from evalscope.api.registry import register_benchmark
from evalscope.constants import ScoringPolicy, Tags
from evalscope.utils.logger import get_logger
from .utils import QUESTION_TYPES, SUBSET_TO_FILE, build_generation_prompt, get_anscheck_prompt

logger = get_logger()


# The official grader prompt ends with "Answer yes or no only."
class AnswerCheck(BaseModel):
    reasoning: str = ''
    verdict: Literal['yes', 'no']


ANSCHECK_CONTRACT = OutputContract(schema_model=AnswerCheck)


@register_benchmark(
    BenchmarkMeta(
        name='longmemeval',
        pretty_name='LongMemEval',
        tags=[Tags.QA, Tags.LONG_CONTEXT, Tags.MULTI_TURN, Tags.RETRIEVAL],
        description="""
## Overview

LongMemEval evaluates long-term interactive memory in chat assistants. Each question is answered from a timestamped multi-session user-assistant history.

## Task Description

- **Task Type**: Long-context / retrieval-log question answering
- **Input**: Multiple dated chat sessions + current question date + question
- **Output**: Free-form answer grounded in the history
- **Subsets**: `s` (~115K-token histories), `m` (~500 sessions, large), and `oracle` (evidence sessions only)

## Key Features

- Covers single-session, multi-session, temporal reasoning, knowledge update, preference, and abstention questions
- Supports full-history long-context prompts and official retrieval-log prompts
- Uses LongMemEval's LLM judge prompts for semantic answer correctness
- Downloads only the selected JSON file from ModelScope

## Evaluation Notes

- Default subset is `s` with `eval_mode=long_context` for standard long-context evaluation
- Use `subset_list=['oracle']` and `extra_params.eval_mode='oracle_context'` for evidence-only upper-bound evaluation
- The `m` subset is large and must be requested explicitly
- `retrieval_log` mode consumes official LongMemEval retrieval logs; it does not run embedding retrieval itself
""",
        dataset_id='evalscope/longmemeval-cleaned',
        metric_list=['accuracy'],
        primary_metric=MetricSelector(name='accuracy', aggregation='mean', dimensions={'scope': 'overall'}),
        few_shot_num=0,
        train_split=None,
        eval_split='test',
        subset_list=['s'],
        default_subset='s',
        extra_params={
            'eval_mode': {
                'type': 'str',
                'description': 'Evaluation mode: oracle_context, long_context, or retrieval_log.',
                'value': 'long_context',
                'choices': ['oracle_context', 'long_context', 'retrieval_log'],
            },
            'retrieval_log_path': {
                'type': 'str | null',
                'description': 'Official LongMemEval retrieval log path for eval_mode=retrieval_log.',
                'value': None,
            },
            'retriever_type': {
                'type': 'str',
                'description': 'Retrieval prompt shape for retrieval_log mode.',
                'value': 'flat-session',
                'choices': ['flat-session', 'flat-turn'],
            },
            'history_format': {
                'type': 'str',
                'description': 'History rendering format.',
                'value': 'json',
                'choices': ['json', 'nl'],
            },
            'user_only': {
                'type': 'bool',
                'description': 'Whether to keep only user turns in history.',
                'value': False,
            },
            'reading_method': {
                'type': 'str',
                'description': 'Prompt reading method. `con` asks the model to extract and reason before answering.',
                'value': 'con',
                'choices': ['direct', 'con'],
            },
            'topk_context': {
                'type': 'int',
                'description': 'Maximum number of history sessions or retrieved chunks included in the prompt.',
                'value': 1000,
            },
        },
    )
)
class LongMemEvalAdapter(DefaultDataAdapter):
    """Adapter for LongMemEval."""
    scoring_policy = ScoringPolicy.JUDGE_ONLY

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.eval_mode = self.extra_params.get('eval_mode', 'long_context')
        self.retrieval_log_path = self.extra_params.get('retrieval_log_path')
        self.retriever_type = self.extra_params.get('retriever_type', 'flat-session')
        self.history_format = self.extra_params.get('history_format', 'json')
        self.user_only = self.extra_params.get('user_only', False)
        self.reading_method = self.extra_params.get('reading_method', 'con')
        self.topk_context = self.extra_params.get('topk_context', 1000)

    def load(self) -> Tuple[DatasetDict, None]:
        self._validate_params()
        if self.eval_mode == 'retrieval_log':
            dataset = self._load_retrieval_log()
            return DatasetDict({'retrieval_log': dataset}), None

        subset_dict = {}
        for subset in self.subset_list:
            records = self._load_subset_records(subset)
            dataset = build_dataset_from_records(
                records=records,
                sample_fields=self.record_to_sample,
                name=f'longmemeval_{subset}',
                location=self.dataset_id,
                limit=self.limit,
                repeats=self.repeats,
                shuffle=self.shuffle,
                seed=self.seed,
            )
            subset_dict[subset] = dataset
        return DatasetDict(subset_dict), None

    def _validate_params(self) -> None:
        if self.eval_mode == 'oracle_context':
            invalid_subsets = [subset for subset in self.subset_list if subset != 'oracle']
            if invalid_subsets:
                raise ValueError('eval_mode=oracle_context only supports subset_list=["oracle"].')
        elif self.eval_mode == 'long_context':
            invalid_subsets = [
                subset for subset in self.subset_list if subset == 'oracle' or subset not in SUBSET_TO_FILE
            ]
            if invalid_subsets:
                raise ValueError('eval_mode=long_context supports subset_list containing only "s" and/or "m".')
        elif self.eval_mode == 'retrieval_log':
            if not self.retrieval_log_path:
                raise ValueError('extra_params["retrieval_log_path"] is required for eval_mode=retrieval_log.')
        else:
            raise ValueError(f'Unsupported eval_mode: {self.eval_mode}')

        if self.history_format not in ['json', 'nl']:
            raise ValueError(f'Unsupported history_format: {self.history_format}')
        if self.reading_method not in ['direct', 'con']:
            raise ValueError(f'Unsupported reading_method: {self.reading_method}')
        if not isinstance(self.topk_context, int) or self.topk_context <= 0:
            raise ValueError('topk_context must be a positive integer.')

    def _load_subset_records(self, subset: str) -> List[Dict[str, Any]]:
        if subset not in SUBSET_TO_FILE:
            raise ValueError(f'Unsupported LongMemEval subset: {subset}. Available subsets: {sorted(SUBSET_TO_FILE)}')
        file_path = self._resolve_dataset_file(SUBSET_TO_FILE[subset])
        records = json.loads(Path(file_path).read_text(encoding='utf-8'))
        return records

    def _resolve_dataset_file(self, file_name: str) -> str:
        if Path(self.dataset_id).exists():
            file_path = Path(self.dataset_id) / file_name
            if not file_path.exists():
                raise FileNotFoundError(f'LongMemEval data file not found: {file_path}')
            return str(file_path)
        return DatasetHub(
            data_id_or_path=self.dataset_id,
            data_source=self.dataset_hub,
            force_redownload=self.force_redownload,
            cache_dir=self.dataset_dir,
        ).download_file(file_name)

    def _load_retrieval_log(self) -> MemoryDataset:
        records = self._read_json_or_jsonl(self.retrieval_log_path)
        return build_dataset_from_records(
            records=records,
            sample_fields=self.record_to_sample,
            name='longmemeval_retrieval_log',
            location=self.retrieval_log_path,
            limit=self.limit,
            repeats=self.repeats,
            shuffle=self.shuffle,
            seed=self.seed,
        )

    @staticmethod
    def _read_json_or_jsonl(path: Union[str, Path]) -> List[Dict[str, Any]]:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f'LongMemEval retrieval log not found: {path}')
        content = path.read_text(encoding='utf-8')
        try:
            data = json.loads(content)
            if isinstance(data, list):
                return data
        except json.JSONDecodeError:
            pass

        try:
            records = [json.loads(line) for line in content.splitlines() if line.strip()]
        except json.JSONDecodeError as e:
            raise ValueError(f'Failed to parse LongMemEval data as JSON or JSONL: {path}') from e
        if records:
            return records
        raise ValueError(f'LongMemEval data must be a list of records: {path}')

    def record_to_sample(self, record: Dict[str, Any]) -> Sample:
        prompt, prompt_metadata = build_generation_prompt(
            entry=record,
            eval_mode=self.eval_mode,
            topk_context=self.topk_context,
            history_format=self.history_format,
            user_only=self.user_only,
            reading_method=self.reading_method,
            retriever_type=self.retriever_type,
        )
        metadata = {
            'question_id': record['question_id'],
            'question_type': record['question_type'],
            'question': record['question'],
            'answer': str(record['answer']),
            'question_date': record['question_date'],
            'answer_session_ids': record.get('answer_session_ids', []),
            'is_abstention': '_abs' in record['question_id'],
            'eval_mode': self.eval_mode,
            **prompt_metadata,
        }
        return Sample(input=[ChatMessageUser(content=prompt)], target=str(record['answer']), metadata=metadata)

    def judge_definition(self, context: JudgeContext) -> JudgeDefinition:

        def request(case, placement, completed_cases, judge_context) -> JudgeRequest:
            metadata = judge_context.task_state.metadata
            prompt = get_anscheck_prompt(
                task=metadata['question_type'],
                question=metadata['question'],
                answer=judge_context.reference,
                response=judge_context.filtered_prediction,
                abstention=metadata.get('is_abstention', False),
            ) + case.output_contract.instruction()
            return JudgeRequest(messages=[ChatMessageUser(content=prompt)])

        def reduce(case_verdicts, judge_context) -> ReducedVerdict:
            return ReducedVerdict(value={'accuracy': 1.0 if case_verdicts[0].value.verdict == 'yes' else 0.0})

        def finalize(score, review, judge_context) -> Score:
            metadata = judge_context.task_state.metadata
            score.metadata.update({
                'question_type': metadata['question_type'],
                'is_abstention': metadata.get('is_abstention', False)
            })
            return score

        return JudgeDefinition.workflow(
            cases=[JudgeCase(case_id='anscheck', output_contract=ANSCHECK_CONTRACT)],
            request=request,
            reduce=reduce,
            main_score_name='accuracy',
            finalize=finalize
        )

    def aggregate_scores(self, sample_scores: List[SampleScore]) -> List[AggScore]:
        valid_scores = [s for s in sample_scores if s.score and 'accuracy' in s.score.value]
        if not valid_scores:
            return []

        agg_scores = [
            self._make_agg_score(metric_name='accuracy', dimensions={'scope': 'overall'}, scores=valid_scores),
        ]

        task_means = []
        for question_type in QUESTION_TYPES:
            type_scores = [s for s in valid_scores if (s.sample_metadata or {}).get('question_type') == question_type]
            if not type_scores:
                continue
            agg = self._make_agg_score(
                metric_name='accuracy', dimensions={'question_type': question_type}, scores=type_scores
            )
            agg_scores.append(agg)
            task_means.append(agg.score)

        if task_means:
            agg_scores.append(
                AggScore(
                    metric_name='accuracy',
                    aggregation='macro_mean',
                    dimensions={'scope': 'question_types'},
                    score=sum(task_means) / len(task_means),
                    num=len(task_means),
                )
            )

        abstention_scores = [s for s in valid_scores if (s.sample_metadata or {}).get('is_abstention')]
        if abstention_scores:
            agg_scores.append(
                self._make_agg_score(
                    metric_name='accuracy', dimensions={'question_type': 'abstention'}, scores=abstention_scores
                )
            )

        return agg_scores

    @staticmethod
    def _make_agg_score(metric_name: str, dimensions: Dict[str, Any], scores: List[SampleScore]) -> AggScore:
        values = [float(s.score.value['accuracy']) for s in scores]
        ids = [s.sample_id for s in scores]
        return AggScore(
            metric_name=metric_name,
            aggregation='mean',
            dimensions=dimensions,
            score=sum(values) / len(values),
            num=len(values),
            ids=ids,
        )
