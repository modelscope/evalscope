import re
from typing import Any, Dict

from evalscope.api.benchmark import BenchmarkMeta, VisionLanguageAdapter
from evalscope.api.dataset import Sample
from evalscope.api.evaluator import TaskState
from evalscope.api.messages import ChatMessageUser, Content, ContentImage, ContentText
from evalscope.api.registry import register_benchmark
from evalscope.constants import Tags
from evalscope.utils.io_utils import bytes_to_base64
from evalscope.utils.logger import get_logger
from evalscope.utils.multi_choices import parse_answers

logger = get_logger()

SUBSET_LIST = ['val']

MULT_CHOICE_PROMPT =r"""
Answer the following multiple choice question. 
The last line of your response should be of the following format: 
'ANSWER: $LETTER' (without quotes) 
where LETTER is one of {letters}. Think step by step before answering.

{question}
""".strip()

DESCRIPTION_TEXT = (
    "As shown in the figure below, existing benchmarks lack"
    "consideration of the vision dependency of evaluation"
    "samples and potential data leakage from"
    "LLMs' and LVLMs' training data."
)
@register_benchmark(
    BenchmarkMeta(
        name='mm_star',
        pretty_name='MMStar',
        tags=[Tags.MULTI_MODAL, Tags.KNOWLEDGE, Tags.QA],
        description=DESCRIPTION_TEXT,
        dataset_id='evalscope/MMStar',
        subset_list=SUBSET_LIST,
        metric_list=['acc'],
        eval_split='val',
        prompt_template=MULT_CHOICE_PROMPT,
    )
)
class MMStarAdapter(VisionLanguageAdapter):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def extract_options(self, text):
        match = re.search(r'Options:\s*(.*)', text, re.DOTALL)
        if not match:
            return ''
        options_content = match.group(1)
        # 提取所有选项标识符（A:、B:、C: 等）
        pattern = r'(?:^|(?<=,))\s*([A-Z]):'
        letters = re.findall(pattern, options_content)
        return ','.join(letters)
    
    def record_to_sample(self, record: Dict[str, Any]) -> Sample:
        letters = self.extract_options(record['question'])
        input_text = MULT_CHOICE_PROMPT.format(
            letters=letters,
            question=record['question']
        )
        content_list: list[Content] = [ContentText(text=input_text)]
        image = record.get('image')
        if image:
            image_base64 = bytes_to_base64(image['bytes'], format='jpeg', add_header=True)
            content_list.append(ContentImage(image=image_base64))
        label_answer = record.get('answer')
        return Sample(
            input=[ChatMessageUser(content=content_list)],
            target=label_answer,
            metadata = {
                'index': record.get('index'),
                'category': record.get('category'),
                'l2_category': record.get('l2_category'),
                'source': record.get('meta_info',{}).get('source'),
                'split': record.get('meta_info',{}).get('split'),
                'image_path': record.get('meta_info',{}).get('image_path')
            }
        )
    
    def extract_answer(self, prediction: str, task_state: TaskState) -> str:
        answers = parse_answers(task_state)
        return ''.join(sorted(list(answers)))