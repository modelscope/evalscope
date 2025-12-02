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
        name='docvqa',
        pretty_name='DocVQA',
        tags=[Tags.MULTI_MODAL, Tags.KNOWLEDGE, Tags.QA],
        description=
        'DocVQA (Document Visual Question Answering) is a benchmark designed to evaluate AI systems on their ability to answer questions based on the content of document images, such as scanned pages, forms, or invoices. Unlike general visual question answering, it requires understanding not just the text extracted by OCR, but also the complex layout, structure, and visual elements of a document.',  # noqa: E501
        dataset_id='lmms-lab/DocVQA',
        subset_list=['DocVQA'],
        metric_list=['anls'],
        eval_split='validation',
        prompt_template=PROMPT,
    )
)
class DocVQAAdapter(VisionLanguageAdapter):

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
                'question_types': record.get('question_types'),
                'docId': record.get('docId'),
                'ucsf_document_id': record.get('ucsf_document_id'),
                'ucsf_document_page_no': record.get('ucsf_document_page_no'),
            }
        )

    def extract_answer(self, prediction: str, task_state: TaskState) -> str:
        import re

        pattern = r'ANSWER:\s*(.*)'
        match = re.search(pattern, prediction)
        if match:
            return match.group(1).strip()
        return prediction.strip()
