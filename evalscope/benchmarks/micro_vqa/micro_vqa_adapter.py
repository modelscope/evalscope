from typing import Any, Dict, List

from evalscope.api.benchmark import BenchmarkMeta, MultiChoiceAdapter, VisionLanguageAdapter
from evalscope.api.dataset import Sample
from evalscope.api.messages import ChatMessageUser, Content, ContentImage, ContentText
from evalscope.api.registry import register_benchmark
from evalscope.constants import Tags
from evalscope.utils.io_utils import bytes_to_base64
from evalscope.utils.logger import get_logger
from evalscope.utils.multi_choices import MultipleChoiceTemplate, answer_character, prompt

logger = get_logger()

MULT_CHOICE_PROMPT = MultipleChoiceTemplate.SINGLE_ANSWER_COT


@register_benchmark(
    BenchmarkMeta(
        name='micro_vqa',
        pretty_name='MicroVQA',
        dataset_id='evalscope/MicroVQA',
        tags=[Tags.KNOWLEDGE, Tags.MULTIPLE_CHOICE, Tags.MULTI_MODAL, Tags.MEDICAL],
        description=
        'MicroVQA is expert-curated benchmark for multimodal reasoning for microscopy-based scientific research',
        metric_list=['acc'],
        eval_split='test',
        prompt_template=MULT_CHOICE_PROMPT,
    )
)
class MicroVQAAdapter(VisionLanguageAdapter, MultiChoiceAdapter):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def record_to_sample(self, record: Dict[str, Any]) -> Sample:
        question: str = record.get('question', '')

        answers_list: List[str] = record['choices']
        content_list: List[Content] = []
        input_text = prompt(question=question, choices=answers_list, template=self.prompt_template)
        content_list.append(ContentText(text=input_text))

        images = record.get('images_list')
        if len(images) > 0:
            for image in images:
                image_base64 = bytes_to_base64(image['bytes'], format='png', add_header=True)
                content_list.append(ContentImage(image=image_base64))

        target = answer_character(record['correct_index'])

        metadata: Dict[str, Any] = {
            'key_question': record.get('key_question'),
            'key_image': record.get('key_image'),
            'correct_answer': record.get('correct_answer'),
            'question_0': record.get('question_0'),
            'answer_0': record.get('answer_0'),
            'comments_0': record.get('comments_0'),
            'incorrect_answer_0': record.get('incorrect_answer_0'),
            'question_1': record.get('question_1'),
            'choices_1': record.get('choices_1'),
            'correct_index_1': record.get('correct_index_1'),
            'question_2': record.get('question_2'),
            'choices_2': record.get('choices_2'),
            'correct_index_2': record.get('correct_index_2'),
            'question_3': record.get('question_3'),
            'choices_3': record.get('choices_3'),
            'correct_index_3': record.get('correct_index_3'),
            'task': record.get('task'),
            'task_str': record.get('task_str'),
            'context_image_generation': record.get('context_image_generation'),
            'context_motivation': record.get('context_motivation'),
            'images_source': record.get('images_source'),
            'image_caption': record.get('image_caption'),
            'key_person': record.get('key_person'),
        }

        return Sample(
            input=[ChatMessageUser(content=content_list)],
            choices=answers_list,
            target=target,
            metadata=metadata,
        )
