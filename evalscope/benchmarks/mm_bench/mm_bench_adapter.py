from typing import Any, Dict

from evalscope.api.benchmark import BenchmarkMeta, VisionLanguageAdapter
from evalscope.api.dataset import Sample
from evalscope.api.evaluator import TaskState
from evalscope.api.messages import ChatMessageUser, Content, ContentImage, ContentText
from evalscope.api.registry import register_benchmark
from evalscope.constants import Tags
from evalscope.utils.io_utils import bytes_to_base64
from evalscope.utils.logger import get_logger
from evalscope.utils.multi_choices import MultipleChoiceTemplate, parse_answers, prompt

logger = get_logger()

SUBSET_LIST_CC = ['cc']
SUBSET_LIST = ['cn', 'en']

MULT_CHOICE_PROMPT = MultipleChoiceTemplate.SINGLE_ANSWER_COT

DESCRIPTION_TEXT = (
    'MMBench is collected from multiple sources,'
    'including public datasets and Internet, '
    'and currently, contains 2974 multiple-choice questions,'
    'covering 20 ability dimensions.'
)


@register_benchmark(
    BenchmarkMeta(
        name='mm_bench_cc',
        pretty_name='MMBench_CC',
        tags=[Tags.MULTI_MODAL, Tags.KNOWLEDGE, Tags.QA],
        description=DESCRIPTION_TEXT,
        dataset_id='lmms-lab/MMBench',
        subset_list=SUBSET_LIST_CC,
        metric_list=['acc'],
        eval_split='test',
        prompt_template=MULT_CHOICE_PROMPT,
    )
)
class MMBenchCCAdapter(VisionLanguageAdapter):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def record_to_sample(self, record: Dict[str, Any]) -> Sample:
        answers_list: list[str] = [record.get('A', ''), record.get('B', ''), record.get('C', ''), record.get('D', '')]
        input_text = prompt(question=record['question'], choices=answers_list, template=MULT_CHOICE_PROMPT)
        content_list: list[Content] = [ContentText(text=input_text)]
        image = record.get('image')
        if image:
            image_base64 = bytes_to_base64(image['bytes'], format='jpeg', add_header=True)
            content_list.append(ContentImage(image=image_base64))
        label_answer = record.get('answer')
        return Sample(
            input=[ChatMessageUser(content=content_list)],
            choices=answers_list,
            target=label_answer,
            metadata={
                'index': record.get('index'),
                'category': record.get('category'),
                'source': record.get('source')
            }
        )

    def extract_answer(self, prediction: str, task_state: TaskState) -> str:
        answers = parse_answers(task_state)
        return ''.join(sorted(list(answers)))


@register_benchmark(
    BenchmarkMeta(
        name='mm_bench',
        pretty_name='MMBench',
        tags=[Tags.MULTI_MODAL, Tags.KNOWLEDGE, Tags.QA],
        description=DESCRIPTION_TEXT,
        dataset_id='lmms-lab/MMBench',
        subset_list=SUBSET_LIST,
        metric_list=['acc'],
        eval_split='dev',
        prompt_template=MULT_CHOICE_PROMPT,
    )
)
class MMBenchAdapter(VisionLanguageAdapter):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def record_to_sample(self, record: Dict[str, Any]) -> Sample:
        answers_list: list[str] = [record.get('A', ''), record.get('B', ''), record.get('C', ''), record.get('D', '')]
        question_hint = record['hint'] + record['question']
        input_text = prompt(question=question_hint, choices=answers_list, template=MULT_CHOICE_PROMPT)
        content_list: list[Content] = [ContentText(text=input_text)]
        image = record.get('image')
        if image:
            image_base64 = bytes_to_base64(image['bytes'], format='jpeg', add_header=True)
            content_list.append(ContentImage(image=image_base64))
        label_answer = record.get('answer')
        return Sample(
            input=[ChatMessageUser(content=content_list)],
            choices=answers_list,
            target=label_answer,
            metadata={
                'index': record.get('index'),
                'category': record.get('category'),
                'source': record.get('source'),
                'L2-category': record.get('L2-category'),
                'comment': record.get('comment'),
                'split': record.get('record')
            }
        )

    def extract_answer(self, prediction: str, task_state: TaskState) -> str:
        answers = parse_answers(task_state)
        return ''.join(sorted(list(answers)))
