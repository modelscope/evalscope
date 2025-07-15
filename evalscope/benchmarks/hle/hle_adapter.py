import re
from collections import defaultdict
from typing import Any, List

from evalscope.benchmarks import Benchmark, DataAdapter
from evalscope.metrics import DEFAULT_PROMPT_TEMPLATE, LLMJudge, exact_match, mean
from evalscope.utils.logger import get_logger

# flake8: noqa

logger = get_logger()

SUBSET_LIST = [
    'Biology/Medicine',
    'Chemistry',
    'Computer Science/AI',
    'Engineering',
    'Humanities/Social Science',
    'Math',
    'Physics',
    'Other',
]


@Benchmark.register(
    name='hle',
    pretty_name="Humanity's-Last-Exam",
    tags=['Knowledge', 'QA'],
    description=
    'Humanity\'s Last Exam (HLE) is a language model benchmark consisting of 2,500 questions across a broad range of subjects. It was created jointly by the Center for AI Safety and Scale AI. The benchmark classifies the questions into the following broad subjects: mathematics (41%), physics (9%), biology/medicine (11%), humanities/social science (9%), computer science/artificial intelligence (10%), engineering (4%), chemistry (7%), and other (9%). Around 14% of the questions require the ability to understand both text and images, i.e., multi-modality. 24% of the questions are multiple-choice; the rest are short-answer, exact-match questions.',  # noqa: E501
    dataset_id='cais/hle',
    subset_list=SUBSET_LIST,
    metric_list=['AverageAccuracy'],
    few_shot_num=0,
    train_split=None,
    eval_split='test',
    prompt_template='{query}\n\nPlease reason step by step, and put your final answer within \\boxed{{}}.',
)
class HLEAdapter(DataAdapter):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.llm_as_a_judge = True

    def load(self, **kwargs):
        kwargs['subset_list'] = ['default']
        data_dict = super().load(**kwargs)
        return self.reformat_subset(data_dict, subset_key='category', format='{}')

    def gen_prompt(self, input_d: dict, subset_name: str, few_shot_list: list, **kwargs) -> dict:
        # remove image preview
        input_d.pop('image_preview', None)
        input_d.pop('rationale_image', None)
        # generate prompt
        question = input_d['question']
        prompt = self.prompt_template.format(query=question)
        image = input_d.get('image', None)
        # build messages for multi-modal input
        messages = []
        if self.system_prompt:
            messages.append({'role': 'system', 'content': self.system_prompt})
        if image:
            messages.append({
                'role':
                'user',
                'content': [{
                    'type': 'text',
                    'text': prompt
                }, {
                    'type': 'image_url',
                    'image_url': {
                        'url': image
                    }
                }]
            })
        else:
            messages.append({'role': 'user', 'content': prompt})
        return self.gen_prompt_data(prompt='', messages=messages)

    def get_gold_answer(self, input_d: dict) -> str:
        return input_d['answer']

    def parse_pred_result(self, result: str, raw_input_d: dict = None, **kwargs) -> str:
        # Extract the answer from the model output \boxed{answer}
        match = re.search(r'\\boxed{([^}]*)}', result)
        if match:
            return match.group(1).strip()
        else:
            logger.warning(f'No answer found in the model output: {result}')
            return ''

    def llm_parse_pred_result(self, result, raw_input_d=None, **kwargs) -> str:
        return result.strip()

    def match(self, gold: str, pred: str) -> dict:
        # simple match
        return {
            'AverageAccuracy': 1.0 if exact_match(gold, pred) else 0.0,
        }

    def llm_match(self, gold: Any, pred: Any, judge: LLMJudge, **kwargs) -> dict:
        raw_input = kwargs.get('raw_input', None)
        question = raw_input['question']
        # get grading response
        prompt = judge.build_prompt(pred, gold, question)
        judge_response = judge(prompt)
        score = judge.get_score(judge_response)
        return {
            'AverageAccuracy': score,
            'response': judge_response,
        }

    def compute_metric(self, review_res_list: List[dict], **kwargs) -> List[dict]:
        # zip dict answers
        res_dict = super().compute_dict_metric(review_res_list, **kwargs)

        return super().compute_metric(res_dict, **kwargs)
