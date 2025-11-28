import re
from typing import Any, Dict, List

from evalscope.api.benchmark import BenchmarkMeta, MultiChoiceAdapter, VisionLanguageAdapter
from evalscope.api.dataset import Sample
from evalscope.api.messages import ChatMessageUser, Content, ContentImage, ContentText
from evalscope.api.registry import register_benchmark
from evalscope.constants import Tags
from evalscope.utils.io_utils import bytes_to_base64
from evalscope.utils.logger import get_logger
from evalscope.utils.multi_choices import format_letter_choices

logger = get_logger()

MULT_CHOICE_PROMPT = r"""
Answer the following multiple choice question. The last line of your response should be of the following format:
'ANSWER: [LETTER]' (without quotes) where [LETTER] is one of {letters}.

{question}
""".strip()

SUBSET_LIST = [
    'Art_Style', 'Counting', 'Forensic_Detection', 'Functional_Correspondence', 'IQ_Test', 'Jigsaw',
    'Multi-view_Reasoning', 'Object_Localization', 'Relative_Depth', 'Relative_Reflectance', 'Semantic_Correspondence',
    'Spatial_Relation', 'Visual_Correspondence', 'Visual_Similarity'
]


@register_benchmark(
    BenchmarkMeta(
        name='blink',
        pretty_name='BLINK',
        tags=[Tags.MULTI_MODAL, Tags.KNOWLEDGE, Tags.MULTIPLE_CHOICE],
        description=
        'BLINK is a benchmark designed to evaluate the core visual perception abilities of multimodal large language models (MLLMs). It transforms 14 classic computer vision tasks into 3,807 multiple-choice questions, accompanied by single or multiple images and visual prompts.',  # noqa: E501
        dataset_id='evalscope/BLINK',
        subset_list=SUBSET_LIST,
        metric_list=['acc'],
        eval_split='val',
        prompt_template=MULT_CHOICE_PROMPT,
    )
)
class BLINKAdapter(VisionLanguageAdapter, MultiChoiceAdapter):
    MAX_IMAGES: int = 4

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def record_to_sample(self, record: Dict[str, Any]) -> Sample:
        choices = record.get('choices')
        input_text = MULT_CHOICE_PROMPT.format(question=record['prompt'], letters=format_letter_choices(choices))
        content_list: List[Content] = [ContentText(text=input_text)]

        for i in range(1, self.MAX_IMAGES + 1):
            image = record.get(f'image_{i}')
            if image:
                image_base64 = bytes_to_base64(image['bytes'], format='jpeg', add_header=True)
                content_list.append(ContentImage(image=image_base64))

        label_answer = record['answer'].strip('(').strip(')')
        return Sample(input=[ChatMessageUser(content=content_list)], choices=choices, target=label_answer)
