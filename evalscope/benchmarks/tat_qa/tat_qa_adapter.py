# TAT-QA adapter: question answering over the hybrid tabular and textual content of financial reports.
#
# Paper: https://aclanthology.org/2021.acl-long.254/ (ACL 2021)
# Dataset: https://modelscope.cn/datasets/evalscope/TAT-QA (mirror of the original release, CC BY 4.0)
# Official evaluation: https://github.com/NExTplusplus/TAT-QA (tatqa_eval.py, MIT License)

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

from evalscope.api.benchmark import BenchmarkMeta, DefaultDataAdapter
from evalscope.api.dataset import DatasetDict, DatasetHub, Sample, build_dataset_from_records
from evalscope.api.evaluator import TaskState
from evalscope.api.metric import Score
from evalscope.api.registry import register_benchmark
from evalscope.constants import Tags

from .utils import SPAN_SEPARATOR, parse_response, score_question

SUBSET_LIST = ['table', 'text', 'table-text']
TEST_FILE = 'tatqa_dataset_test_gold.json'
# Pinned revisions: the ModelScope mirror and the original Hugging Face release (test set with gold answers).
DATASET_REVISION = '8d64c441239133dd8042c28e0afeb52e09887b5c'
HF_DATASET_REVISION = 'c96247f5077eac447f63527fd3dcfdc58bb56d6a'

PROMPT_TEMPLATE = """Below is a table and some paragraphs from a financial report, followed by a question.

Table:
{table}

Paragraphs:
{paragraphs}

Question: {question}

Answer the question using the table and the paragraphs:
- If the answer is text from the table or the paragraphs, copy it exactly. If there are several, separate them with " | ".
- If the answer is a number, give the number only and put its scale (thousand, million, billion or percent) on the Scale line.
Think step by step, then end your response with these two lines:
Answer: <answer>
Scale: <none, thousand, million, billion or percent>"""  # noqa: E501

DESCRIPTION = """
## Overview

TAT-QA (Tabular And Textual dataset for Question Answering) evaluates numerical reasoning over hybrid
financial data: each question is asked about a table from a real financial report together with at least
two paragraphs describing it. Answers are text spans, sets of spans, counts or numbers that must be
computed with arithmetic, and most carry a scale (thousand, million, billion or percent).

## Task Description

- **Task Type**: Table + text question answering with numerical reasoning
- **Input**: A table, its paragraphs and a question
- **Output**: Reasoning, then an `Answer:` line and a `Scale:` line
- **Domain**: Financial reports

## Key Features

- 1,663 test questions over 277 tables, with the gold answers released by the authors
- Subsets by where the answer comes from: `table` (736), `text` (381) and `table-text` (546)
- Answer types: span (714), arithmetic (699), multi-span (210) and count (40)
- Scales: none (836), thousand (319), percent (294) and million (214)

## Evaluation Notes

- Metrics are the official exact match (`exact_match`) and numeracy-focused F1 (`f1`, primary), computed with a port of
  `tatqa_metric.py` that gives the same per-question scores as the official script
- Numbers are compared after applying the scale and rounding, so `1.2` with scale `million` and `1,200,000`
  match; the official F1 equals EM for arithmetic and count questions
- Multiple spans are separated by `|` on the `Answer:` line; an unknown or missing scale counts as no scale
- Questions are answered zero-shot from the full table and paragraphs (no retrieval)
- Data is mirrored on ModelScope (`evalscope/TAT-QA`); the original release is on Hugging Face (`next-tat/TAT-QA`)
- [Paper](https://aclanthology.org/2021.acl-long.254/) | [Code](https://github.com/NExTplusplus/TAT-QA) |
  [Dataset](https://huggingface.co/datasets/next-tat/TAT-QA) | [Leaderboard](https://nextplusplus.github.io/TAT-QA/)
"""


def format_table(rows: List[List[str]]) -> str:
    return '\n'.join(' | '.join(str(cell).strip() for cell in row) for row in rows)


def format_paragraphs(paragraphs: List[Dict[str, Any]]) -> str:
    return '\n'.join(p['text'] for p in sorted(paragraphs, key=lambda p: p['order']))


@register_benchmark(
    BenchmarkMeta(
        name='tat_qa',
        pretty_name='TAT-QA',
        dataset_id='evalscope/TAT-QA',
        dataset_revision=DATASET_REVISION,
        hf_dataset_id='next-tat/TAT-QA',
        hf_dataset_revision=HF_DATASET_REVISION,
        paper_url='https://aclanthology.org/2021.acl-long.254/',
        tags=[Tags.REASONING, Tags.QA, Tags.MATH],
        description=DESCRIPTION,
        subset_list=SUBSET_LIST,
        eval_split='test',
        few_shot_mode='disabled',
        metric_list=['exact_match', 'f1'],
        primary_metric='f1',
        prompt_template=PROMPT_TEMPLATE,
        evaluation_version='v1.0',
    )
)
class TATQAAdapter(DefaultDataAdapter):
    """Data adapter for TAT-QA, scored with the official EM / F1."""

    def load(self) -> Tuple[DatasetDict, None]:
        records = [
            {'table': doc['table']['table'], 'paragraphs': doc['paragraphs'], **qa}
            for doc in self._read_documents()
            for qa in doc['questions']
        ]
        subsets = {}
        for subset in self.subset_list:
            subsets[subset] = build_dataset_from_records(
                records=[r for r in records if r['answer_from'] == subset],
                sample_fields=self.record_to_sample,
                name=f'tat_qa_{subset}',
                location=self.dataset_id,
                limit=self.limit,
                repeats=self.repeats,
                shuffle=self.shuffle,
                seed=self.seed,
            )
        return DatasetDict(subsets), None

    def _read_documents(self) -> List[Dict[str, Any]]:
        # The repository holds one JSON file per split rather than a `datasets` config, so the test file is
        # read directly.
        if Path(self.dataset_id).is_dir():
            path = Path(self.dataset_id) / TEST_FILE
        else:
            path = DatasetHub(
                data_id_or_path=self.dataset_id,
                data_source=self.dataset_hub,
                revision=self.dataset_revision,
                force_redownload=self.force_redownload,
                cache_dir=self.dataset_dir,
            ).download_file(TEST_FILE)
        with open(path, encoding='utf-8') as f:
            return json.load(f)

    def record_to_sample(self, record: Dict[str, Any]) -> Sample:
        answer = record['answer']
        target = SPAN_SEPARATOR.join(map(str, answer)) if isinstance(answer, list) else str(answer)
        return Sample(
            input=self.prompt_template.format(
                table=format_table(record['table']),
                paragraphs=format_paragraphs(record['paragraphs']),
                question=record['question'],
            ),
            target=target,
            subset_key=record['answer_from'],
            metadata={
                'uid': record['uid'],
                'answer': answer,
                'answer_type': record['answer_type'],
                'answer_from': record['answer_from'],
                'scale': record['scale'],
            },
        )

    def format_prompt_template(self, sample: Sample) -> str:
        # The prompt is fully built in `record_to_sample`.
        return sample.input

    def extract_answer(self, prediction: str, task_state: TaskState) -> str:
        spans, scale = parse_response(prediction)
        return json.dumps({'answer': spans, 'scale': scale}, ensure_ascii=False)

    def match_score(
        self, original_prediction: str, filtered_prediction: str, reference: str, task_state: TaskState
    ) -> Score:
        parsed = json.loads(filtered_prediction)
        em, f1 = score_question(task_state.metadata, parsed['answer'], parsed['scale'])
        return Score(
            extracted_prediction=filtered_prediction,
            prediction=original_prediction,
            value={'exact_match': em, 'f1': f1},
            main_score_name='f1',
            metadata={'answer_type': task_state.metadata['answer_type'], 'parsed': bool(parsed['answer'])},
        )
