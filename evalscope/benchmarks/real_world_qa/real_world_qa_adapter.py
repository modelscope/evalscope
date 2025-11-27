import re
from typing import Any, Dict, List

from evalscope.api.benchmark import BenchmarkMeta, VisionLanguageAdapter
from evalscope.api.dataset import Sample
from evalscope.api.evaluator import TaskState
from evalscope.api.messages import ChatMessageUser, Content, ContentImage, ContentText
from evalscope.api.registry import register_benchmark
from evalscope.constants import Tags
from evalscope.utils.io_utils import bytes_to_base64
from evalscope.utils.logger import get_logger

logger = get_logger()

SUBSET_LIST = ['default']

OPEN_PROMPT = (
    'Read the picture and solve the following problem step by step.'
    'The last line of your response should be of the form'
    ' "ANSWER: [ANSWER]" (without quotes) where [ANSWER] is the answer to the problem.\n\n'
    '{question}\n\n'
    'Remember to put your answer on its own line at the end in the form'
    ' "ANSWER: [ANSWER]" (without quotes) where [ANSWER] is the answer to the problem,'
    ' and you do not need to use a \\boxed command.'
)


@register_benchmark(
    BenchmarkMeta(
        name='real_world_qa',
        pretty_name='RealWorldQA',
        tags=[Tags.MULTI_MODAL, Tags.KNOWLEDGE, Tags.QA],
        description=
        'RealWorldQA is a benchmark designed to evaluate the real-world spatial understanding capabilities of multimodal AI models, contributed by XAI. It assesses how well these models comprehend physical environments. The benchmark consists of 700+ images, each accompanied by a question and a verifiable answer. These images are drawn from real-world scenarios, including those captured from vehicles. The goal is to advance AI models\' understanding of our physical world.',  # noqa: E501
        dataset_id='lmms-lab/RealWorldQA',
        subset_list=SUBSET_LIST,
        metric_list=['acc'],
        eval_split='test',
        prompt_template=OPEN_PROMPT,
    )
)
class RealWorldQAAdapter(VisionLanguageAdapter):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def record_to_sample(self, record: Dict[str, Any]) -> Sample:
        content_list: list[Content] = [ContentText(text=OPEN_PROMPT.format(question=record['question']))]
        image = record.get('image')
        if image:
            image_base64 = bytes_to_base64(image['bytes'], format='webp', add_header=True)
            content_list.append(ContentImage(image=image_base64))
        return Sample(
            input=[ChatMessageUser(content=content_list)],
            target=record['answer'],
            metadata={'image_path': record['image_path']}
        )

    def extract_answer(self, prediction: str, task_state: TaskState) -> str:
        pattern = r'ANSWER:\s*(.*)'
        match = re.search(pattern, prediction)
        if match:
            return match.group(1).strip()
        return ''
