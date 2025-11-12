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
        name='science_qa',
        pretty_name='ScienceQA',
        dataset_id='AI-ModelScope/ScienceQA',
        tags=[Tags.KNOWLEDGE, Tags.QA, Tags.MULTIPLE_CHOICE, Tags.MULTI_MODAL],
        description=
        'Learn to Explain: Multimodal Reasoning via Thought Chains for Science Question Answering',
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
class ScienceQAAdapter(VisionLanguageAdapter):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def record_to_sample(self, record: Dict[str, Any]) -> Sample:
        question: str = record.get('question', '')
        # answers_list: List[str] = ast.literal_eval(record['choices'])
        answers_list: List[str] = record['choices']
        content_list: list[Content] = []
        input_text = prompt(question=question, choices=answers_list, template=MULT_CHOICE_PROMPT)
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
            target=target,
            metadata=metadata,
        )
    
    def extract_answer(self, prediction: str, task_state: TaskState) -> str:
        answers = parse_answers(task_state)
        return ''.join(sorted(list(answers)))