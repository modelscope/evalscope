import ast
import re
from typing import Any, Dict, List

from evalscope.api.benchmark import BenchmarkMeta, VisionLanguageAdapter
from evalscope.api.dataset import Sample
from evalscope.api.evaluator import TaskState
from evalscope.api.messages import ChatMessageUser, Content, ContentImage, ContentText
from evalscope.api.metric import Score
from evalscope.api.registry import register_benchmark
from evalscope.constants import Tags
from evalscope.utils.io_utils import bytes_to_base64
from evalscope.utils.logger import get_logger
from evalscope.utils.multi_choices import MultipleChoiceTemplate, answer_character, parse_answers_zh, prompt

logger = get_logger()

SUBSET_LIST = ['biology', 'chemistry', 'geography', 'history', 'math', 'physics', 'politics']

MULT_CHOICE_PROMPT = MultipleChoiceTemplate.CHINESE_SINGLE_ANSWER_TEMPLATE_COT
MULTIPLE_RESPONSE_PROMPT = MultipleChoiceTemplate.CHINESE_MULTIPLE_ANSWER_TEMPLATE_COT

FILL_IN_BLANK_PROMPT = """逐步解决以下填空问题。这包括多个问题，按照"问题(1): $QUESTION_1 问题(2): $QUESTION_2 问题(3): $QUESTION_3..."的格式（不含引号）$QUESTION-X（X指一个数字）代表一个具体的问题

{question}

记住在最后单独一行写上你的答案，格式为"答案(X): [ANSWER_X]"（不含引号），其中[ANSWER_X]是对应X号问题(QUESTION-X)的答案。
"""  # noqa: E501

MULTI_CHOICE_TYPE = 'multiple-choice'
MULTIPLE_RESPONSE_TYPE = 'multiple-response'
FILL_IN_BLANK_TYPE = 'fill-in-the-blank'


@register_benchmark(
    BenchmarkMeta(
        name='cmmu',
        pretty_name='CMMU',
        dataset_id='evalscope/CMMU',
        tags=[Tags.KNOWLEDGE, Tags.QA, Tags.MULTIPLE_CHOICE, Tags.MULTI_MODAL],
        description=(
            'CMMU is a novel multi-modal benchmark designed to evaluate domain-specific '
            'knowledge across seven foundational subjects: math, biology, physics, chemistry, '
            'geography, politics, and history.'
        ),
        subset_list=SUBSET_LIST,
        metric_list=[{
            'acc': {
                'numeric': True
            }
        }],
        default_subset='default',
        eval_split='val',
        prompt_template=MULT_CHOICE_PROMPT,
    )
)
class CMMUAdapter(VisionLanguageAdapter):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.reformat_subset = True
        self._use_llm_judge = True

    def record_to_sample(self, record: Dict[str, Any]) -> Sample:
        content_list, answers_list = CMMUAdapter.create_content_and_answers_list(record)

        metadata: Dict[str, Any] = {
            'type': record.get('type'),
            'grade_band': record.get('grade_band'),
            'difficulty': record.get('difficulty'),
            'split': record.get('split'),
            'subject': record.get('subject'),
            'sub_questions': record.get('sub_questions'),
            'solution_info': record.get('solution_info'),
            'id': record.get('id'),
        }

        question_type = record.get('type')
        if question_type == FILL_IN_BLANK_TYPE:
            target_str = '\n'.join(f'答案 ({i+1}): {ans}' for i, ans in enumerate(record['answer']))
            return Sample(
                input=[ChatMessageUser(content=content_list)],
                target=target_str,
                subset_key=record['subject'],
                metadata=metadata,
            )
        else:
            if len(record['answer']) > 0:
                target_str = record['answer'][0]
            else:
                target_str = ''
            return Sample(
                input=[ChatMessageUser(content=content_list)],
                target=target_str,
                choices=answers_list,
                subset_key=record['subject'],
                metadata=metadata,
            )

    def extract_answer(self, prediction, task_state):
        question_type = task_state.metadata['type']
        if question_type in [MULTI_CHOICE_TYPE, MULTIPLE_RESPONSE_TYPE]:
            is_multi_answer = question_type == MULTIPLE_RESPONSE_TYPE
            answers = parse_answers_zh(task_state, multiple_correct=is_multi_answer)
            multi_answer = ''.join(sorted(list(answers)))
            return multi_answer
        else:
            return prediction.strip()

    def llm_match_score(
        self,
        original_prediction: str,
        filtered_prediction: str,
        reference: str,
        task_state: TaskState,
    ) -> Score:
        score = Score(
            extracted_prediction=filtered_prediction,
            prediction=original_prediction,
        )

        question = task_state.input_text
        question_type = task_state.metadata['type']
        if question_type in [MULTI_CHOICE_TYPE, MULTIPLE_RESPONSE_TYPE]:
            score.value = {'acc': 1 if filtered_prediction == reference else 0}
        else:
            from .prompt import EVALUATION_SYSTEM_PROMPT, EVALUATION_USER_TEMPLATE

            prompt = EVALUATION_USER_TEMPLATE.format(
                question=question, target=reference, predicted_answer=original_prediction
            )
            judge_response = self.llm_judge.judge(prompt, system_prompt=EVALUATION_SYSTEM_PROMPT)
            try:
                ans_parsed = ast.literal_eval(judge_response)
                correctness = ans_parsed.get('correct', 0)
                analysis = ans_parsed.get('analysis', '')
            except Exception:
                pattern = re.compile(r'"correct"\s*:\s*1')
                match = re.search(pattern, judge_response)
                correctness = 1 if match else 0

            score.value = {
                'acc': correctness,
            }
            score.explanation = f'LLM judge: {judge_response}'
            score.metadata = {
                'source': 'llm_judge',
                'judge_strategy': self.judge_strategy,
                'analysis': analysis,
                'model': self.llm_judge.model_id
            }
        score.main_score_name = 'acc'
        return score

    @staticmethod
    def create_content_and_answers_list(record: Dict[str, Any]) -> tuple[List[Content], List[str]]:
        """
        Create a list of content elements and a list of answers from a record.

        Args:
            record (dict): The record containing question, images, and options.


        Returns:
            tuple: A tuple containing:
                - content_list (list): A list of content elements (text and images).
                - answers_list (list): A list of possible answers (for multiple-choice questions).
        """
        question_type = record['type']

        if question_type == MULTI_CHOICE_TYPE:
            answers_list: List[str] = record['options']
            input_text = prompt(question=record['question_info'], choices=answers_list, template=MULT_CHOICE_PROMPT)
            content_list: List[Content] = [ContentText(text=input_text)]
        elif question_type == MULTIPLE_RESPONSE_TYPE:
            answers_list: List[str] = record['options']
            input_text = prompt(
                question=record['question_info'], choices=answers_list, template=MULTIPLE_RESPONSE_PROMPT
            )
            content_list: List[Content] = [ContentText(text=input_text)]
        else:
            answers_list: List[str] = []
            sub_questions_str = '\n'.join(f'问题 ({i+1}): {q}' for i, q in enumerate(record['sub_questions']))
            open_question = record['question_info'] + sub_questions_str
            content_list: List[Content] = [ContentText(text=FILL_IN_BLANK_PROMPT.format(question=open_question))]

        image = record.get('image')
        if image:
            image_base64 = bytes_to_base64(image['bytes'], format='png', add_header=True)
            content_list.append(ContentImage(image=image_base64))

        return content_list, answers_list
