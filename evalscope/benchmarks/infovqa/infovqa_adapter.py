import json
from typing import Any, Dict, List

from evalscope.api.benchmark import BenchmarkMeta, VisionLanguageAdapter
from evalscope.api.dataset import Sample
from evalscope.api.evaluator.state import TaskState
from evalscope.api.messages import ChatMessageUser, Content, ContentImage, ContentText
from evalscope.api.registry import register_benchmark
from evalscope.constants import Tags
from evalscope.utils.io_utils import bytes_to_base64
from evalscope.utils.logger import get_logger

logger = get_logger()

PROMPT = """Answer the question according to the image using a single word or phrase.
{question}
The last line of your response should be of the form "ANSWER: [ANSWER]" (without quotes) where [ANSWER] is the answer to the question."""  # noqa: E501


@register_benchmark(
    BenchmarkMeta(
        name='infovqa',
        pretty_name='InfoVQA',
        tags=[Tags.MULTI_MODAL, Tags.KNOWLEDGE, Tags.QA],
        description="""
## Overview

InfoVQA (Infographic Visual Question Answering) is a benchmark designed to evaluate AI models' ability to answer questions based on information-dense images such as charts, graphs, diagrams, maps, and infographics. It focuses on understanding complex visual information presentations.

## Task Description

- **Task Type**: Infographic Question Answering
- **Input**: Infographic image + natural language question
- **Output**: Single word or phrase answer
- **Domains**: Data visualization, information graphics, visual reasoning

## Key Features

- Focuses on information-dense visual content
- Covers charts, graphs, diagrams, maps, and infographics
- Requires understanding visual layouts and data representations
- Tests information extraction and reasoning abilities
- Questions vary in complexity from direct lookup to inference

## Evaluation Notes

- Default evaluation uses the **validation** split
- Primary metric: **ANLS** (Average Normalized Levenshtein Similarity)
- Answers should be in format "ANSWER: [ANSWER]"
- Includes OCR text extraction as metadata for analysis
- Uses same dataset source as DocVQA (InfographicVQA subset)
""",  # noqa: E501
        dataset_id='lmms-lab/DocVQA',
        subset_list=['InfographicVQA'],
        metric_list=['anls'],
        eval_split='validation',
        prompt_template=PROMPT,
    )
)
class InfoVQAAdapter(VisionLanguageAdapter):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_aggregation_name = False

    def record_to_sample(self, record: Dict[str, Any]) -> Sample:

        input_text = PROMPT.format(question=record['question'])
        content_list: List[Content] = [ContentText(text=input_text)]
        image = record.get('image')
        if image:
            image_base64 = bytes_to_base64(image['bytes'], format='png', add_header=True)
            content_list.append(ContentImage(image=image_base64))
        return Sample(
            input=[ChatMessageUser(content=content_list)],
            target=json.dumps(record.get('answers')),  # answers is a list
            metadata={
                'questionId': record.get('questionId'),
                'answer_type': record.get('answer_type'),
                'image_url': record.get('image_url'),
                'ocr': record.get('ocr'),
            }
        )

    def extract_answer(self, prediction: str, task_state: TaskState) -> str:
        import re

        pattern = r'ANSWER:\s*(.*)'
        match = re.search(pattern, prediction)
        if match:
            return match.group(1).strip()
        return prediction.strip()
