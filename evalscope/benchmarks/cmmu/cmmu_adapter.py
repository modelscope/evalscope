import ast
import re
from typing import Any, Dict, List

from evalscope.api.benchmark import BenchmarkMeta, VisionLanguageAdapter
from evalscope.api.dataset import Sample
from evalscope.api.evaluator import TaskState
from evalscope.api.messages import ChatMessageUser, Content, ContentImage, ContentText
from evalscope.api.registry import register_benchmark
from evalscope.api.metric import Score
from evalscope.constants import Tags
from evalscope.utils.io_utils import bytes_to_base64
from evalscope.utils.logger import get_logger
from evalscope.utils.multi_choices import MultipleChoiceTemplate, parse_answers, answer_character, prompt

# flake8: noqa

logger = get_logger()

SUBSET_LIST = [
    'biology',
    'chemistry',
    'geography',
    'history',
    'math',
    'physics',
    'politics'
]

MULT_CHOICE_PROMPT = MultipleChoiceTemplate.SINGLE_ANSWER_COT
MULTIPLE_RESPONSE_PROMPT = MultipleChoiceTemplate.MULTIPLE_ANSWER_COT
OPEN_PROMPT = """
Solve the following fill in the blank problem step by step. 
This includes multiple questions according to "QUESTION_1: $QUESTION_1 QUESTION_2: $QUESTION_2 QUESTION_3: $QUESTION_3..."(without quotes)
$QUESTION-X (X refers to a number) represents a specific question
The answer only needs to be answered by filling in the blanks and separating them with semicolons
Answer in Chinese
The last line of your response should be of the form "ANSWER_X: $ANSWER_X" (without quotes) where $ANSWER_X is the answer to QUESTION-X corresponding to the X number.

{question}

Remember to put your answer on its own line at the end in the form "ANSWER_X: $ANSWER_X" (without quotes) where $ANSWER_X is the answer to QUESTION-X corresponding to the X number, and you do not need to use a \\boxed command.
"""
GRADER_TEMPLATE = """
You are an expert evaluator specializing in assessing fill-in-the-blank questions and choice questions in primary school to hight school exams. I will give you a question, the expected correct answer, and a test-taker's response to the question.
Here is a new example.
```
question: {question}
the expected correct answer: {target}
a test-taker's response to the question: {predicted_answer}
```
You need to understand the given question, compare the standard answer with the provided response, and output the following values:
Return Y for correct, N for incorrect.

Just return the letters "Y" or "N", with no text around it.
"""

MULTI_CHOICE_TYPE = 'multiple-choice'
MULTIPLE_RESPONSE_TYPE = 'multiple-response'
OPEN_TYPE = 'fill-in-the-blank'

@register_benchmark(
    BenchmarkMeta(
        name='cmmu',
        pretty_name='CMMU',
        dataset_id='evalscope/CMMU',
        tags=[Tags.KNOWLEDGE, Tags.QA, Tags.MULTIPLE_CHOICE, Tags.MULTI_MODAL],
        description=(
            'CMMU is a novel multi-modal benchmark designed to evaluate domain-specific'
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
        question: str = record.get('question_info', '')
        content_list, answers_list = CMMUAdapter.create_content_and_answers_list(record)
        
        if len(record['answer'])>0:
            target = record['answer'][0]
        else:
            target = ""
        
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
        if question_type==OPEN_TYPE:
            target_str="" 
            if len(record['answer'])>0:
                for i in range(len(record['answer'])):
                    target_str += (";" if target_str else "") + f"ANSWER_{i}: {record['answer'][i]}"
            return Sample(
                input=[ChatMessageUser(content=content_list)],
                target=target_str,
                subset_key=record['subject'],
                metadata=metadata,
            )
        else:
            if len(record['answer'])>0:
                target_str = record['answer'][0]
            else:
                target_str = ""
            return Sample(
                input=[ChatMessageUser(content=content_list)],
                target=target_str,
                choices=answers_list,
                subset_key=record['subject'],
                metadata=metadata,
            )
    
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
        if question_type in [MULTI_CHOICE_TYPE,MULTIPLE_RESPONSE_TYPE]:
            answers = parse_answers(task_state)
            multi_answer = ''.join(sorted(list(answers)))
            prompt = GRADER_TEMPLATE.format(question=question, target=reference, predicted_answer=multi_answer)
        else:
            prompt = GRADER_TEMPLATE.format(question=question, target=reference, predicted_answer=filtered_prediction)
        judge_response = self.llm_judge.judge(prompt)
        match = re.search(r'(Y|N)', judge_response)
        res = match.group(0) if match else 'N'

        score.value = {
            'is_correct': 1 if res == 'Y' else 0,
        }
        score.explanation = f'LLM judge: {judge_response}'
        score.metadata = {
            'source': 'llm_judge',
            'judge_strategy': self.judge_strategy,
            'model': self.llm_judge.model_id
        }
        score.main_score_name = 'is_correct'
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

        if question_type==MULTI_CHOICE_TYPE:
            answers_list: List[str] = record['options']
            input_text = prompt(question=record['question_info'], choices=answers_list, template=MULT_CHOICE_PROMPT)
            content_list: List[Content] = [ContentText(text=input_text)]
        elif question_type == MULTIPLE_RESPONSE_TYPE:
            answers_list: List[str] = record['options']
            input_text = prompt(question=record['question_info'], choices=answers_list, template=MULTIPLE_RESPONSE_PROMPT)
            content_list: List[Content] = [ContentText(text=input_text)]
        else:
            answers_list: List[str] = []
            sub_questions_str = ""
            if len(record['sub_questions'])>0:
                for i in range(len(record['sub_questions'])):
                    sub_questions_str += (";" if sub_questions_str else "") + f"QUESTION_{i}: {record['sub_questions'][i]}"
            open_question = record['question_info'] + sub_questions_str
            content_list: List[Content] = [ContentText(text=OPEN_PROMPT.format(question=open_question))]
        
        image = record.get('image')
        if image:
            image_base64 = bytes_to_base64(image['bytes'], format='png', add_header=True)
            content_list.append(ContentImage(image=image_base64))
        
        return content_list, answers_list