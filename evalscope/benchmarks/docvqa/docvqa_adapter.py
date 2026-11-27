import json
from typing import Any, Dict, List

from evalscope.api.benchmark import BenchmarkMeta, VisionLanguageAdapter
from evalscope.api.dataset import Sample
from evalscope.api.evaluator.state import TaskState
from evalscope.api.messages import ChatMessageUser, Content, ContentText
from evalscope.api.registry import register_benchmark
from evalscope.constants import Tags
from evalscope.utils.logger import get_logger

logger = get_logger()

PROMPT = """Answer the question according to the image using a single word or phrase.
{question}
The last line of your response should be of the form "ANSWER: [ANSWER]" (without quotes) where [ANSWER] is the answer to the question."""  # noqa: E501


@register_benchmark(
    BenchmarkMeta(
        name='docvqa',
        pretty_name='DocVQA',
        tags=[Tags.MULTI_MODAL, Tags.KNOWLEDGE, Tags.QA],
        description="""
## Overview

DocVQA (Document Visual Question Answering) is a benchmark designed to evaluate AI systems' ability to answer questions based on document images such as scanned pages, forms, invoices, and reports. It requires understanding complex document layouts, structure, and visual elements beyond simple text extraction.

## Task Description

- **Task Type**: Document Visual Question Answering
- **Input**: Document image + natural language question
- **Output**: Single word or phrase answer extracted from document
- **Domains**: Document understanding, OCR, layout comprehension

## Key Features

- Covers diverse document types (forms, invoices, letters, reports)
- Requires understanding document layout and structure
- Tests both text extraction and contextual reasoning
- Questions require locating and interpreting specific information
- Combines OCR capabilities with visual understanding

## Evaluation Notes

- Default evaluation uses the **validation** split
- Primary metric: **ANLS** (Average Normalized Levenshtein Similarity)
- Answers should be in format "ANSWER: [ANSWER]"
- ANLS metric accounts for minor OCR/spelling variations
- Multiple valid answers may be accepted for each question
""",  # noqa: E501
        dataset_id='lmms-lab/DocVQA',
        hf_dataset_id='lmms-lab/DocVQA',
        subset_list=['DocVQA'],
        metric_list=['anls'],
        eval_split='validation',
        prompt_template=PROMPT,
        evaluation_version='v1.2',
    )
)
class DocVQAAdapter(VisionLanguageAdapter):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def record_to_sample(self, record: Dict[str, Any]) -> Sample:

        template = self.prompt_template if self.prompt_template is not None else PROMPT
        input_text = template.format(question=record['question'])
        content_list: List[Content] = [ContentText(text=input_text)]
        image = record.get('image')
        if image:
            image_payload = self._normalize_media_value(image, media_type='image')
            content_list.append(self._content_image_from_value(image_payload))
        return Sample(
            input=[ChatMessageUser(content=content_list)],
            target=json.dumps(record.get('answers')),  # answers is a list
            metadata={
                'questionId': record.get('questionId'),
                'question_types': record.get('question_types'),
                'docId': record.get('docId'),
                'ucsf_document_id': record.get('ucsf_document_id'),
                'ucsf_document_page_no': record.get('ucsf_document_page_no'),
            },
        )

    def extract_answer(self, prediction: str, task_state: TaskState) -> str:
        from evalscope.utils.multi_choices import extract_answer_line

        answer = extract_answer_line(prediction)
        return answer if answer is not None else prediction.strip()
