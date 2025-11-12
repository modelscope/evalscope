from typing import Any, Dict

from evalscope.api.benchmark import BenchmarkMeta, MultiChoiceAdapter
from evalscope.api.dataset import Sample
from evalscope.api.registry import register_benchmark
from evalscope.constants import Tags
from evalscope.utils.logger import get_logger
from evalscope.utils.multi_choices import MultipleChoiceTemplate

logger = get_logger()

SUBJECT_MAPPING = {
    'abstract_algebra': ['Abstract Algebra', 'math', 'STEM'],
    'anatomy': ['Anatomy', 'health', 'Other'],
    'astronomy': ['Astronomy', 'physics', 'STEM'],
    'business_ethics': ['Business Ethics', 'business', 'Other'],
    'clinical_knowledge': ['Clinical Knowledge', 'health', 'Other'],
    'college_biology': ['College Biology', 'biology', 'STEM'],
    'college_chemistry': ['College Chemistry', 'chemistry', 'STEM'],
    'college_computer_science': ['College Computer Science', 'computer science', 'STEM'],
    'college_mathematics': ['College Mathematics', 'math', 'STEM'],
    'college_medicine': ['College Medicine', 'health', 'Other'],
    'college_physics': ['College Physics', 'physics', 'STEM'],
    'computer_security': ['Computer Security', 'computer science', 'STEM'],
    'conceptual_physics': ['Conceptual Physics', 'physics', 'STEM'],
    'econometrics': ['Econometrics', 'economics', 'Social Science'],
    'electrical_engineering': ['Electrical Engineering', 'engineering', 'STEM'],
    'elementary_mathematics': ['Elementary Mathematics', 'math', 'STEM'],
    'formal_logic': ['Formal Logic', 'philosophy', 'Humanities'],
    'global_facts': ['Global Facts', 'other', 'Other'],
    'high_school_biology': ['High School Biology', 'biology', 'STEM'],
    'high_school_chemistry': ['High School Chemistry', 'chemistry', 'STEM'],
    'high_school_computer_science': ['High School Computer Science', 'computer science', 'STEM'],
    'high_school_european_history': ['High School European History', 'history', 'Humanities'],
    'high_school_geography': ['High School Geography', 'geography', 'Social Science'],
    'high_school_government_and_politics': ['High School Government And Politics', 'politics', 'Social Science'],
    'high_school_macroeconomics': ['High School Macroeconomics', 'economics', 'Social Science'],
    'high_school_mathematics': ['High School Mathematics', 'math', 'STEM'],
    'high_school_microeconomics': ['High School Microeconomics', 'economics', 'Social Science'],
    'high_school_physics': ['High School Physics', 'physics', 'STEM'],
    'high_school_psychology': ['High School Psychology', 'psychology', 'Social Science'],
    'high_school_statistics': ['High School Statistics', 'math', 'STEM'],
    'high_school_us_history': ['High School Us History', 'history', 'Humanities'],
    'high_school_world_history': ['High School World History', 'history', 'Humanities'],
    'human_aging': ['Human Aging', 'health', 'Other'],
    'human_sexuality': ['Human Sexuality', 'culture', 'Social Science'],
    'international_law': ['International Law', 'law', 'Humanities'],
    'jurisprudence': ['Jurisprudence', 'law', 'Humanities'],
    'logical_fallacies': ['Logical Fallacies', 'philosophy', 'Humanities'],
    'machine_learning': ['Machine Learning', 'computer science', 'STEM'],
    'management': ['Management', 'business', 'Other'],
    'marketing': ['Marketing', 'business', 'Other'],
    'medical_genetics': ['Medical Genetics', 'health', 'Other'],
    'miscellaneous': ['Miscellaneous', 'other', 'Other'],
    'moral_disputes': ['Moral Disputes', 'philosophy', 'Humanities'],
    'moral_scenarios': ['Moral Scenarios', 'philosophy', 'Humanities'],
    'nutrition': ['Nutrition', 'health', 'Other'],
    'philosophy': ['Philosophy', 'philosophy', 'Humanities'],
    'prehistory': ['Prehistory', 'history', 'Humanities'],
    'professional_accounting': ['Professional Accounting', 'other', 'Other'],
    'professional_law': ['Professional Law', 'law', 'Humanities'],
    'professional_medicine': ['Professional Medicine', 'health', 'Other'],
    'professional_psychology': ['Professional Psychology', 'psychology', 'Social Science'],
    'public_relations': ['Public Relations', 'politics', 'Social Science'],
    'security_studies': ['Security Studies', 'politics', 'Social Science'],
    'sociology': ['Sociology', 'culture', 'Social Science'],
    'us_foreign_policy': ['Us Foreign Policy', 'politics', 'Social Science'],
    'virology': ['Virology', 'health', 'Other'],
    'world_religions': ['World Religions', 'philosophy', 'Humanities'],
}

SUBSET_LIST = list(SUBJECT_MAPPING.keys())


@register_benchmark(
    BenchmarkMeta(
        name='mmlu_redux',
        pretty_name='MMLU-Redux',
        tags=[Tags.MULTIPLE_CHOICE, Tags.KNOWLEDGE],
        description=
        'MMLU-Redux is a benchmark for evaluating language models on multiple-choice questions across various subjects. It includes questions from different domains, where the model must select the correct answer from given options. '  # noqa: E501
        'The bad answers are corrected.',  # noqa: E501
        dataset_id='AI-ModelScope/mmlu-redux-2.0',
        subset_list=SUBSET_LIST,
        metric_list=[{
            'acc': {
                'allow_inclusion': True
            }
        }],
        few_shot_num=0,
        train_split=None,
        eval_split='test',
        prompt_template=MultipleChoiceTemplate.SINGLE_ANSWER_COT,
    )
)
class MMLUReduxAdapter(MultiChoiceAdapter):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        if self.few_shot_num > 0:
            self.few_shot_num = 0
            logger.warning('Few-shot examples are not supported for MMLU-Redux dataset. Setting few_shot_num to 0.')

    def record_to_sample(self, record: Dict[str, Any]) -> Sample:
        error_type = record['error_type']
        choices = record['choices']
        target_index_list = [int(record['answer'])]
        correct_answer = record['correct_answer']
        if error_type == 'no_correct_answer' and correct_answer:
            choices[target_index_list[0]] = correct_answer
        elif error_type == 'wrong_groundtruth' and correct_answer:
            try:
                target_index_list = [int(correct_answer)]
            except ValueError:
                choice_index = ord(correct_answer) - ord('A')
                target_index_list = [choice_index]
        elif error_type == 'multiple_correct_answers' and correct_answer:
            correct_answer = correct_answer.strip('()')
            try:
                correct_answer = correct_answer.replace(' and ', ',').replace(' or ', ',')
                target_index_list = list(map(int, correct_answer.split(',')))
            except ValueError:
                try:
                    target_index_list = [ord(c) - ord('A') for c in correct_answer.split(',')]
                except TypeError:
                    # find the index of the correct answer in choices
                    target_index_list = [choices.index(c) for c in correct_answer.split(',') if c in choices]

        return Sample(
            input=record['question'],
            choices=choices,
            target=['ABCD'[i] for i in target_index_list] if target_index_list else ['A', 'B', 'C', 'D'],
            metadata={
                'error_type': error_type,
                'correct_answer': correct_answer,
                'potential_reason': record.get('potential_reason', ''),
            },
        )
