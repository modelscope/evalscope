from typing import Any, Dict, List

from evalscope.api.benchmark import BenchmarkMeta, MultiChoiceAdapter, VisionLanguageAdapter
from evalscope.api.dataset import Sample
from evalscope.api.evaluator import TaskState
from evalscope.api.messages import ChatMessageUser, Content, ContentImage, ContentText
from evalscope.api.registry import register_benchmark
from evalscope.constants import Tags
from evalscope.utils.io_utils import bytes_to_base64
from evalscope.utils.logger import get_logger
from evalscope.utils.multi_choices import MultipleChoiceTemplate, answer_character, parse_answers, prompt

logger = get_logger()

MULT_CHOICE_PROMPT = MultipleChoiceTemplate.SINGLE_ANSWER_COT


@register_benchmark(
    BenchmarkMeta(
        name='science_qa',
        pretty_name='ScienceQA',
        dataset_id='AI-ModelScope/ScienceQA',
        tags=[Tags.KNOWLEDGE, Tags.MULTIPLE_CHOICE, Tags.MULTI_MODAL],
        description=
        'ScienceQA is a multimodal benchmark consisting of multiple-choice science questions derived from elementary and high school curricula. It covers a diverse range of subjects, including natural science, social science, and language science. A key feature of this benchmark is that most questions are accompanied by both image and text contexts, and are annotated with detailed lectures and explanations that support the correct answer, facilitating research into models that can generate chains of thought.',  # noqa: E501
        metric_list=['acc'],
        eval_split='test',
        prompt_template=MULT_CHOICE_PROMPT,
    )
)
class ScienceQAAdapter(VisionLanguageAdapter, MultiChoiceAdapter):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def record_to_sample(self, record: Dict[str, Any]) -> Sample:
        question: str = record.get('question', '')

        answers_list: List[str] = record['choices']
        content_list: List[Content] = []
        input_text = prompt(question=question, choices=answers_list, template=self.prompt_template)
        content_list.append(ContentText(text=input_text))

        image = record.get('image')
        if image:
            image_base64 = bytes_to_base64(image['bytes'], format='png', add_header=True)
            content_list.append(ContentImage(image=image_base64))

        target = answer_character(record['answer'])

        metadata: Dict[str, Any] = {
            'hint': record.get('hint'),
            'task': record.get('task'),
            'grade': record.get('grade'),
            'subject': record.get('subject'),
            'topic': record.get('topic'),
            'category': record.get('category'),
            'skill': record.get('skill'),
            'lecture': record.get('lecture'),
            'solution': record.get('solution'),
        }

        return Sample(
            input=[ChatMessageUser(content=content_list)],
            choices=answers_list,
            target=target,
            metadata=metadata,
        )
