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

MULT_CHOICE_PROMPT = MultipleChoiceTemplate.SINGLE_ANSWER_COT

@register_benchmark(
    BenchmarkMeta(
        name='a_okvqa',
        pretty_name='A-OKVQA',
        dataset_id='HuggingFaceM4/A-OKVQA',
        tags=[Tags.KNOWLEDGE, Tags.QA, Tags.MULTIPLE_CHOICE, Tags.MULTI_MODAL],
        description=
        'a crowdsourced dataset composed of a diverse set of about 25K questions requiring a broad base of commonsense and world knowledge to answer.',
        subset_list=SUBSET_LIST,
        metric_list=[{
            'acc': {
                'numeric': True
            }
        }],
        default_subset='default',
        train_split='train',
        eval_split='validation',
        prompt_template=MULT_CHOICE_PROMPT,
    )
)
class AOkvqaAdapter(VisionLanguageAdapter):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def record_to_sample(self, record: Dict[str, Any]) -> Sample:
        question: str = record.get('question', '')
        answers_list: List[str] = record['choices']
        content_list: list[Content] = []
        input_text = prompt(question=question, choices=answers_list, template=MULT_CHOICE_PROMPT)
        content_list.append(ContentText(text=input_text))

        image = record.get('image')
        if image:
            image_base64 = bytes_to_base64(image['bytes'], format='jpeg', add_header=True)
            content_list.append(ContentImage(image=image_base64))
        
        target = answer_character(record['correct_choice_idx'])

        metadata: Dict[str, Any] = {
            'question_id': record.get('question_id'),
            'direct_answers': record.get('direct_answers'),
            'difficult_direct_answer': record.get('difficult_direct_answer'),
            'rationales': record.get('rationales'),
        }

        return Sample(
            input=[ChatMessageUser(content=content_list)],
            target=target,
            metadata=metadata,
        )
