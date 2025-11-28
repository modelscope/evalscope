# flake8: noqa: E501
from typing import Any, Dict, List

from evalscope.api.benchmark import BenchmarkMeta, MultiChoiceAdapter, VisionLanguageAdapter
from evalscope.api.dataset import Sample
from evalscope.api.messages import ChatMessageUser, Content, ContentImage, ContentText
from evalscope.api.registry import register_benchmark
from evalscope.constants import Tags
from evalscope.utils.io_utils import bytes_to_base64
from evalscope.utils.logger import get_logger

logger = get_logger()

MULT_CHOICE_PROMPT = """
Answer the following multiple choice question. The last line of your response should be of the following format: 'ANSWER: [LETTER]' (without quotes) where [LETTER] is one of A, B, C, D. Think step by step before answering.

{question}
"""

SUBSET_LIST = [
    'Quantitative Reasoning', 'Other', 'Positional Reasoning', 'Stylistic Reasoning', 'Spatial Reasoning',
    'Attribute Reasoning'
]


@register_benchmark(
    BenchmarkMeta(
        name='visulogic',
        pretty_name='VisuLogic',
        dataset_id='evalscope/VisuLogic',
        tags=[Tags.MATH, Tags.REASONING, Tags.MULTIPLE_CHOICE, Tags.MULTI_MODAL],
        description=
        'VisuLogic is a benchmark aimed at evaluating the visual reasoning capabilities of Multi-modal Large Language Models (MLLMs), independent of textual reasoning processes. It features carefully constructed visual reasoning tasks spanning multiple categories, divided into six types based on required reasoning skills (e.g., Quantitative Reasoning, which involves understanding and deducing changes in the quantity of elements in images). Unlike existing benchmarks, VisuLogic is a challenging visual reasoning benchmark that is inherently difficult to articulate using language, providing a more rigorous evaluation of the visual reasoning capabilities of MLLMs.',
        subset_list=SUBSET_LIST,
        metric_list=['acc'],
        eval_split='test',
        prompt_template=MULT_CHOICE_PROMPT,
    )
)
class VisuLogicAdapter(VisionLanguageAdapter, MultiChoiceAdapter):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.reformat_subset = True

    def record_to_sample(self, record: Dict[str, Any]) -> Sample:
        question = record.get('question', '')
        content_list: List[Content] = []
        prompt_text = self.prompt_template.format(question=question).strip()
        content_list.append(ContentText(text=prompt_text))

        image = record.get('image')
        if image and isinstance(image, dict):
            image_bytes = image.get('bytes')
            if image_bytes:
                image_base64 = bytes_to_base64(image_bytes, format='png', add_header=True)
                content_list.append(ContentImage(image=image_base64))

        metadata = {
            'id': record['id'],
        }

        return Sample(
            input=[ChatMessageUser(content=content_list)],
            target=record['label'],
            choices=['A', 'B', 'C', 'D'],
            subset_key=record['tag'],
            metadata=metadata,
        )
