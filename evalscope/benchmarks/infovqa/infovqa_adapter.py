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
        description=
        'InfoVQA (Information Visual Question Answering) is a benchmark designed to evaluate how well AI models can answer questions based on information-dense images, such as charts, graphs, diagrams, maps, and infographics.',  # noqa: E501
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
