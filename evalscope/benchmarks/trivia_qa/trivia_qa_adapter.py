# Copyright (c) Alibaba, Inc. and its affiliates.
# Copyright (c) EleutherAI Inc, and its affiliates.

from typing import Any, Dict

from evalscope.api.benchmark import BenchmarkMeta, DefaultDataAdapter
from evalscope.api.dataset import Sample
from evalscope.api.evaluator import TaskState
from evalscope.api.metric import Score
from evalscope.api.registry import register_benchmark
from evalscope.constants import Tags
from evalscope.utils.logger import get_logger

logger = get_logger()

PROMPT_TEMPLATE = """
Read the content and answer the following question.

Content: {content}

Question: {question}

Keep your The last line of your response should be of the form "ANSWER: [ANSWER]" (without quotes) where [ANSWER] is the answer to the problem.
""".lstrip()  # noqa: E501


@register_benchmark(
    BenchmarkMeta(
        name='trivia_qa',
        pretty_name='TriviaQA',
        dataset_id='evalscope/trivia_qa',
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
        metric_list=[{
            'acc': {
                'allow_inclusion': True
            }
        }],
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
            input=question, target=answers, metadata={
                'question_id': record['question_id'],
                'content': content
            }
        )

    def format_prompt_template(self, sample):
        return self.prompt_template.format(content=sample.metadata['content'], question=sample.input)

    def extract_answer(self, prediction: str, task_state: TaskState):
        # use regex to extract the answer from the prediction
        import re

        pattern = r'ANSWER:\s*(.*)'
        match = re.search(pattern, prediction)
        if match:
            return match.group(1).strip()
        return prediction.strip()
