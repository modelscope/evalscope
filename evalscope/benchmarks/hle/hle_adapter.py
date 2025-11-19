import re
from typing import Any, Dict, List

from evalscope.api.benchmark import BenchmarkMeta, DefaultDataAdapter
from evalscope.api.dataset import Sample
from evalscope.api.evaluator import TaskState
from evalscope.api.messages import ChatMessage, ChatMessageSystem, ChatMessageUser, Content, ContentImage, ContentText
from evalscope.api.metric import Score
from evalscope.api.registry import register_benchmark
from evalscope.constants import Tags
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

ANSWER_TYPE_EXACT_MATCH = 'exactMatch'
ANSWER_TYPE_MULTIPLE_CHOICE = 'multipleChoice'

# System prompt constants
SYSTEM_EXACT_ANSWER = 'Your response should be in the following format:\nExplanation: {your explanation for your final answer}\nExact Answer: {your succinct, final answer}\nConfidence: {your confidence score between 0% and 100% for your answer}'

SYSTEM_MC = 'Your response should be in the following format:\nExplanation: {your explanation for your answer choice}\nAnswer: {your chosen answer}\nConfidence: {your confidence score between 0% and 100% for your answer}'

JUDGE_PROMPT = """Judge whether the following [response] to [question] is correct or not based on the precise and unambiguous [correct_answer] below.

[question]: {question}

[response]: {response}

[correct_answer]: {correct_answer}

Your judgment must focus only on if there are meaningful differences between [correct_answer] and the [response]. Do not comment on any background to the problem, do not attempt to solve the problem, do not argue for any answer different than [correct_answer], focus only on whether the answers match. Explain why the [response] is correct or incorrect based on [correct_answer] in one or two sentences. Finally, write your answer in the format 'GRADE: C' for correct answer or 'GRADE: I' for incorrect answer.
"""


@register_benchmark(
    BenchmarkMeta(
        name='hle',
        pretty_name="Humanity's-Last-Exam",
        tags=[Tags.KNOWLEDGE, Tags.QA],
        description='Humanity\'s Last Exam (HLE) is a language model benchmark consisting of 2,500 '
        'questions across a broad range of subjects. It was created jointly by the Center '
        'for AI Safety and Scale AI. The benchmark classifies the questions into the '
        'following broad subjects: mathematics (41%), physics (9%), biology/medicine (11%), '
        'humanities/social science (9%), computer science/artificial intelligence (10%), '
        'engineering (4%), chemistry (7%), and other (9%). Around 14% of the questions '
        'require the ability to understand both text and images, i.e., multi-modality. '
        '24% of the questions are multiple-choice; the rest are short-answer, exact-match questions. \n'
        '**To evaluate the performance of model without multi-modality capabilities, '
        'please set the `extra_params["include_multi_modal"]` to `False`.**',  # noqa: E501
        dataset_id='cais/hle',
        subset_list=SUBSET_LIST,
        metric_list=['acc'],
        eval_split='test',
        prompt_template='{question}',
        extra_params={
            'include_multi_modal': {
                'type': 'bool',
                'description': 'Include multi-modal (image) questions during evaluation.',
                'value': True
            }
        }
    )
)
class HLEAdapter(DefaultDataAdapter):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._use_llm_judge = True  # Use LLM as a judge by default
        self.reformat_subset = True
        self.include_multi_modal = self.extra_params.get('include_multi_modal', True)

    def record_to_sample(self, record: Dict[str, Any]) -> Sample:
        answer_type = record['answer_type']
        system_prompt = (SYSTEM_EXACT_ANSWER if answer_type == ANSWER_TYPE_EXACT_MATCH else SYSTEM_MC)
        text_content = ContentText(text=record['question'])

        content: List[Content] = [text_content]
        if record['image']:
            image_content = ContentImage(image=record['image'])
            content.append(image_content)

        messages: List[ChatMessage] = [
            ChatMessageSystem(content=system_prompt),
            ChatMessageUser(content=content),
        ]
        return Sample(
            input=messages,
            subset_key=record['category'],
            metadata={
                'uid': record['id'],
                'author_name': record['author_name'],
                'rationale': record['rationale'],
                'raw_subject': record['raw_subject'],
                'category': record['category'],
                'has_image': bool(record['image']),
            },
            target=record['answer'],
        )

    def sample_filter(self, sample):
        if not self.include_multi_modal:
            if sample.metadata is not None and sample.metadata['has_image']:
                return False
        return True

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

        confidence = 100
        if task_state.output and task_state.output.completion:
            confidence_match = re.search(r'confidence:\s*(\d+)', task_state.output.completion, re.IGNORECASE)
            if confidence_match:
                confidence = int(confidence_match.group(1))

        judge_prompt = JUDGE_PROMPT.format(
            question=task_state.input_text, response=filtered_prediction, correct_answer=reference
        )

        # Request judge and obtain score
        judge_response = self.llm_judge.judge(prompt=judge_prompt)

        # Parse judge response to get accuracy score
        accuracy_score = re.search(r'GRADE:\s*([CI])', judge_response, re.IGNORECASE)
        if accuracy_score:
            grade = accuracy_score.group(1).upper()
            score.value = {
                'acc': 1.0 if grade == 'C' else 0.0,
            }
        else:
            score.value = {
                'acc': 0.0,
            }
        score.explanation = f'LLM judge: {judge_response}'
        score.metadata = {
            'source': 'llm_judge',
            'judge_strategy': self.judge_strategy,
            'model': self.llm_judge.model_id,
            'confidence': confidence,
        }
        score.main_score_name = 'acc'
        return score
