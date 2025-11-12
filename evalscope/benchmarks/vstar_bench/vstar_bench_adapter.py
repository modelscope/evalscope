from typing import Any, Dict, List

from evalscope.api.benchmark import BenchmarkMeta, VisionLanguageAdapter
from evalscope.api.dataset import Sample
from evalscope.api.evaluator import TaskState
from evalscope.api.messages import ChatMessageUser, Content, ContentImage, ContentText
from evalscope.api.registry import register_benchmark
from evalscope.constants import Tags
from evalscope.utils.io_utils import bytes_to_base64
from evalscope.utils.logger import get_logger
from evalscope.utils.multi_choices import MultipleChoiceTemplate, parse_answers, answer_character, prompt

# flake8: noqa

logger = get_logger()

SUBSET_LIST = ['default']

MULT_CHOICE_PROMPT = """
Answer the following multiple choice question. The last line of your response should be of the following format: 'ANSWER: $LETTER' (without quotes) where LETTER is one of A, B, C, D. Think step by step before answering.

{question}
"""

@register_benchmark(
    BenchmarkMeta(
        name='vstar_bench',
        pretty_name='vstar-bench',
        dataset_id='lmms-lab/vstar-bench',
        tags=[Tags.KNOWLEDGE, Tags.QA, Tags.MULTIPLE_CHOICE, Tags.MULTI_MODAL],
        description=
        'This dataset includes data sources, prompt information (including content and characters), images, reward models (including real labels and styles), and additional information (including correct answers, IDs, multiple-choice options, and original questions).',
        subset_list=SUBSET_LIST,
        metric_list=[{
            'acc': {
                'numeric': True
            }
        }],
        default_subset='default',
        eval_split='test',
        prompt_template=MULT_CHOICE_PROMPT,
    )
)
class VstarBenchAdapter(VisionLanguageAdapter):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def record_to_sample(self, record: Dict[str, Any]) -> Sample:
        question: str = record.get('text', '')
        content_list: List[Content] = []
        prompt_text = MULT_CHOICE_PROMPT.format(question=question).strip()
        content_list.append(ContentText(text=prompt_text))

        image = record.get('image')
        if image:
            image_base64 = bytes_to_base64(image['bytes'], format='jpeg', add_header=True)
            content_list.append(ContentImage(image=image_base64))
        
        target = record.get('label', '')

        metadata: Dict[str, Any] = {
            'category': record.get('category'),
            'question_id': record.get('question_id'),
        }

        return Sample(
            input=[ChatMessageUser(content=content_list)],
            target=target,
            metadata=metadata,
        )
        