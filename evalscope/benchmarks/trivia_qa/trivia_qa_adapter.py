# Copyright (c) Alibaba, Inc. and its affiliates.
# Copyright (c) EleutherAI Inc, and its affiliates.

import re
import string
from typing import Any, Dict

from evalscope.api.benchmark import BenchmarkMeta, DefaultDataAdapter
from evalscope.api.dataset import Sample
from evalscope.api.evaluator import TaskState
from evalscope.api.metric import Score
from evalscope.api.registry import register_benchmark
from evalscope.constants import Tags
from evalscope.utils.logger import get_logger

logger = get_logger()

# Characters the official evaluation (triviaqa_evaluation.py) turns into spaces.
_PUNCTUATION = set(string.punctuation + '‘’´`')


def normalize_answer(text: str) -> str:
    """Normalize an answer as the official TriviaQA evaluation does; ``normalized_aliases`` are in this form."""
    text = ''.join(' ' if ch in _PUNCTUATION else ch for ch in text.replace('_', ' ').lower())
    return ' '.join(re.sub(r'\b(a|an|the)\b', ' ', text).split())


PROMPT_TEMPLATE = """
Read the content and answer the following question.

Content: {content}

Question: {question}

The last line of your response should be of the form "ANSWER: [ANSWER]" (without quotes) where [ANSWER] is the answer to the problem.
""".lstrip()  # noqa: E501


@register_benchmark(
    BenchmarkMeta(
        name='trivia_qa',
        evaluation_version='v1.2',
        pretty_name='TriviaQA',
        dataset_id='evalscope/trivia_qa',
        hf_dataset_id='mandarjoshi/trivia_qa',
        tags=[Tags.QA, Tags.READING_COMPREHENSION],
        description="""
## Overview

TriviaQA is a large-scale reading comprehension dataset containing over 650K question-answer-evidence triples. Questions are collected from trivia enthusiast websites and paired with Wikipedia articles as evidence documents.

## Task Description

- **Task Type**: Reading Comprehension / Question Answering
- **Input**: Question with Wikipedia context passage
- **Output**: Answer extracted or generated from context
- **Domain**: General knowledge trivia questions

## Key Features

- 650K+ question-answer-evidence triples
- Questions written by trivia enthusiasts (naturally challenging)
- Multiple valid answer aliases for flexible evaluation
- Wikipedia articles provide evidence passages
- Tests both reading comprehension and knowledge retrieval

## Evaluation Notes

- Default configuration uses **0-shot** evaluation
- Uses the Wikipedia reading comprehension subset (rc.wikipedia)
- Answers should follow the format: "ANSWER: [ANSWER]"
- Supports inclusion-based matching for answer comparison
- Evaluates on validation split
""",
        subset_list=['rc.wikipedia'],
        few_shot_num=0,
        train_split=None,
        eval_split='validation',
        metric_list=[{'acc': {'allow_inclusion': True}}],
        prompt_template=PROMPT_TEMPLATE,
    )
)
class TriviaQaAdapter(DefaultDataAdapter):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def record_to_sample(self, record: Dict[str, Any]) -> Sample:
        question = record['question']
        answers = record['answer']['aliases'] + record['answer']['normalized_aliases']
        content = record['entity_pages']['wiki_context']
        return Sample(
            input=question, target=answers, metadata={'question_id': record['question_id'], 'content': content}
        )

    def format_prompt_template(self, sample):
        return self.prompt_template.format(content=sample.metadata['content'], question=sample.input)

    def extract_answer(self, prediction: str, task_state: TaskState):
        # use regex to extract the answer from the prediction
        from evalscope.utils.multi_choices import extract_answer_line

        answer = extract_answer_line(prediction)
        return normalize_answer(answer if answer is not None else prediction)
