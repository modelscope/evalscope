# FinQA adapter: numerical reasoning over financial reports, answered with a program in the FinQA DSL.
#
# Paper: https://aclanthology.org/2021.emnlp-main.300/ (EMNLP 2021)
# Code and data: https://github.com/czyssrs/FinQA (MIT License)

from typing import Any, Dict

from evalscope.api.benchmark import BenchmarkMeta, DefaultDataAdapter
from evalscope.api.dataset import Sample
from evalscope.api.evaluator import TaskState
from evalscope.api.metric import Score
from evalscope.api.registry import register_benchmark
from evalscope.constants import Tags

from .utils import parse_program, score_program

# The dataset is a byte-identical mirror of the official repository's `dataset/test.json`
# (pinned commit 0f16e286, sha256 831dbfb2...); the upstream repo is the canonical source.

PROMPT_TEMPLATE = """Below are the text and the table of a financial report, followed by a question.

{context}

Question: {question}

Write a program that computes the answer, using these operations:
- add(a, b), subtract(a, b), multiply(a, b), divide(a, b), exp(a, b) (a to the power b), greater(a, b) (yes if a > b, else no)
- table_max(row, none), table_min(row, none), table_sum(row, none), table_average(row, none): over the numbers in a table row, where row is the row name exactly as in the first column
Arguments are numbers from the report (plain digits, no units, commas or % signs), constants such as const_100 or const_1000 (const_m1 is -1), or #n, the result of step n (counting from 0).
Use one operation per step (no nested operations) and separate steps and arguments with ", ", e.g. "subtract(5829, 5735), divide(#0, 5735)" for (5829 - 5735) / 5735. A percentage is computed as a ratio, without multiplying by 100.
Think step by step, then end your response with a line of the form:
Program: <program>"""  # noqa: E501

DESCRIPTION = """
## Overview

FinQA evaluates numerical reasoning over financial reports: each question is about one page of an S&P 500
company's earnings report (text and a table), and its answer is computed with a short program of
arithmetic and table operations written by financial experts.

## Task Description

- **Task Type**: Numerical reasoning over text and tables
- **Input**: The report text before and after the table, the table and a question
- **Output**: Reasoning, then a `Program:` line in the FinQA DSL, e.g. `subtract(5829, 5735), divide(#0, 5735)`
- **Domain**: Financial reports (earnings reports of S&P 500 companies)

## Key Features

- 1,147 test questions from the official public test set (gold programs and answers included)
- Programs use `add`, `subtract`, `multiply`, `divide`, `exp`, `greater` and the table operations
  `table_max`, `table_min`, `table_sum`, `table_average`; most need one to three steps

## Evaluation Notes

- Metrics follow the official `evaluate.py`: `execution_accuracy` (primary) runs the predicted program on the
  question's table and compares the result exactly with the gold answer, both rounded to 5 decimals;
  `program_accuracy` checks that the program is symbolically equal to the gold program
- The scoring is a port of the official code and gives the same per-question results
- Execution is exact, so a percentage must be computed as a ratio (`divide(#0, 5735)`), as in the gold programs
- Steps and arguments are separated by `, ` as in the dataset; extra whitespace around brackets and separators
  is normalized, and `op(a,b)` is read as `op(a, b)`. Nested operations are invalid, as in the official DSL
- The whole page (text before and after the table and the table) is given, without retrieval
- [Paper](https://aclanthology.org/2021.emnlp-main.300/) | [Code and data](https://github.com/czyssrs/FinQA)
"""


def format_context(record: Dict[str, Any]) -> str:
    table = '\n'.join(' | '.join(str(cell).strip() for cell in row) for row in record['table'])
    pre = '\n'.join(record['pre_text'])
    post = '\n'.join(record['post_text'])
    return f'Text:\n{pre}\n\nTable:\n{table}\n\nText:\n{post}'


@register_benchmark(
    BenchmarkMeta(
        name='finqa',
        pretty_name='FinQA',
        dataset_id='evalscope/finqa',
        dataset_revision='ac65ea606c70f20a4162ed37c04b9779e43e11d0',
        paper_url='https://aclanthology.org/2021.emnlp-main.300/',
        tags=[Tags.REASONING, Tags.QA, Tags.MATH],
        description=DESCRIPTION,
        subset_list=['default'],
        eval_split='test',
        few_shot_mode='disabled',
        metric_list=['execution_accuracy', 'program_accuracy'],
        primary_metric='execution_accuracy',
        prompt_template=PROMPT_TEMPLATE,
        evaluation_version='v1.0',
    )
)
class FinQAAdapter(DefaultDataAdapter):
    """Data adapter for FinQA, scored with the official execution and program accuracy."""

    def record_to_sample(self, record: Dict[str, Any]) -> Sample:
        qa = record['qa']
        answer = qa['exe_ans']
        # Hub loaders can perturb float literals by ~1 ulp (0.10039 -> 0.10039000000000001). The
        # official protocol compares results rounded to 5 decimals and every gold answer has at most
        # 5 decimals, so this normalization is a no-op on byte-exact data and repairs the drift.
        if isinstance(answer, float):
            answer = round(answer, 5)
        return Sample(
            input=qa['question'],
            target=qa['program'],
            metadata={
                'id': record['id'],
                'context': format_context(record),
                'program': qa['program'],
                'answer': answer,
                'table': record['table'],
            },
        )

    def format_prompt_template(self, sample: Sample) -> str:
        return self.prompt_template.format(context=sample.metadata['context'], question=sample.input)

    def extract_answer(self, prediction: str, task_state: TaskState) -> str:
        return parse_program(prediction) or ''

    def match_score(
        self, original_prediction: str, filtered_prediction: str, reference: str, task_state: TaskState
    ) -> Score:
        meta = task_state.metadata
        exe, prog = score_program(filtered_prediction, meta['program'], meta['answer'], meta['table'])
        return Score(
            extracted_prediction=filtered_prediction,
            prediction=original_prediction,
            value={'execution_accuracy': exe, 'program_accuracy': prog},
            main_score_name='execution_accuracy',
            metadata={'gold_answer': meta['answer'], 'parsed': bool(filtered_prediction)},
        )
