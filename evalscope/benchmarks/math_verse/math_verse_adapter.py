# flake8: noqa: E501
from typing import Any, Dict

from evalscope.api.benchmark import BenchmarkMeta, VisionLanguageAdapter
from evalscope.api.dataset import Sample
from evalscope.api.messages import ChatMessageUser, Content, ContentImage, ContentText
from evalscope.api.registry import register_benchmark
from evalscope.constants import Tags
from evalscope.utils.io_utils import bytes_to_base64
from evalscope.utils.logger import get_logger

logger = get_logger()

MULTI_CHOICE_TYPE = 'multi-choice'
OPEN_TYPE = 'free-form'

OPEN_PROMPT = '{question}\nPlease reason step by step, and put your final answer within \\boxed{{}}.'

MULT_CHOICE_PROMPT = """
Answer the following multiple choice question. The last line of your response should be of the following format: 'ANSWER: [LETTER]' (without quotes) where [LETTER] is one of A, B, C, D. Think step by step before answering.

{question}
"""

SUBSET_LIST = ['Text Dominant', 'Text Lite', 'Vision Intensive', 'Vision Dominant', 'Vision Only']


@register_benchmark(
    BenchmarkMeta(
        name='math_verse',
        pretty_name='MathVerse',
        dataset_id='evalscope/MathVerse',
        tags=[Tags.MATH, Tags.REASONING, Tags.MULTIPLE_CHOICE, Tags.MULTI_MODAL],
        description=
        'MathVerse, an all-around visual math benchmark designed for an equitable and in-depth evaluation of MLLMs. 2,612 high-quality, multi-subject math problems with diagrams from publicly available sources. Each problem is then transformed by human annotators into six distinct versions, each offering varying degrees of information content in multi-modality, contributing to 15K test samples in total. This approach allows MathVerse to comprehensively assess whether and how much MLLMs can truly understand the visual diagrams for mathematical reasoning.',
        subset_list=SUBSET_LIST,
        metric_list=[{
            'acc': {
                'numeric': True
            }
        }],
        default_subset='testmini',
        eval_split='testmini',
        prompt_template=OPEN_PROMPT,
    )
)
class MathVerseAdapter(VisionLanguageAdapter):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.reformat_subset = True
        self._use_llm_judge = True

    def record_to_sample(self, record: Dict[str, Any]) -> Sample:
        """
        Convert a dataset record to a Sample. Unifies handling for both multi-choice and free-form.
        Builds the content list inline and appends image content if provided.

        Args:
            record: Raw dataset record.

        Returns:
            Sample: The standardized sample ready for evaluation.
        """
        question_type = record.get('question_type', OPEN_TYPE)
        question: str = record.get('question', '')
        content_list: list[Content] = []

        # Choose prompt text based on type; keep a single unified flow for creating Sample
        if question_type == MULTI_CHOICE_TYPE:
            prompt_text = MULT_CHOICE_PROMPT.format(question=question).strip()
        else:
            prompt_text = OPEN_PROMPT.format(question=question).strip()

        content_list.append(ContentText(text=prompt_text))

        # Append image if exists
        image = record.get('image')
        if image and isinstance(image, dict):
            image_bytes = image.get('bytes')
            if image_bytes:
                image_base64 = bytes_to_base64(image_bytes, format='png', add_header=True)
                content_list.append(ContentImage(image=image_base64))

        metadata: Dict[str, Any] = {
            'sample_index': record.get('sample_index'),
            'problem_index': record.get('problem_index'),
            'problem_version': record.get('problem_version'),
            'question_type': question_type,
            'query_wo': record.get('query_wo'),
            'query_cot': record.get('query_cot'),
            'question_for_eval': record.get('question_for_eval'),
        }

        return Sample(
            input=[ChatMessageUser(content=content_list)],
            target=record['answer'],
            subset_key=record['problem_version'],
            metadata=metadata,
        )

    def extract_answer(self, prediction: str, task_state):
        from evalscope.metrics.math_parser import extract_answer

        return extract_answer(prediction)
