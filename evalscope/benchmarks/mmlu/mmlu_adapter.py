# Copyright (c) Alibaba, Inc. and its affiliates.
import csv
import os

from evalscope.benchmarks import Benchmark, DataAdapter
from evalscope.constants import EvalType, OutputType
from evalscope.metrics import exact_match
from evalscope.utils import ResponseParser
from evalscope.utils.logger import get_logger

# flake8: noqa

logger = get_logger()

SUBSET_LIST = [
    'high_school_european_history',
    'business_ethics',
    'clinical_knowledge',
    'medical_genetics',
    'high_school_us_history',
    'high_school_physics',
    'high_school_world_history',
    'virology',
    'high_school_microeconomics',
    'econometrics',
    'college_computer_science',
    'high_school_biology',
    'abstract_algebra',
    'professional_accounting',
    'philosophy',
    'professional_medicine',
    'nutrition',
    'global_facts',
    'machine_learning',
    'security_studies',
    'public_relations',
    'professional_psychology',
    'prehistory',
    'anatomy',
    'human_sexuality',
    'college_medicine',
    'high_school_government_and_politics',
    'college_chemistry',
    'logical_fallacies',
    'high_school_geography',
    'elementary_mathematics',
    'human_aging',
    'college_mathematics',
    'high_school_psychology',
    'formal_logic',
    'high_school_statistics',
    'international_law',
    'high_school_mathematics',
    'high_school_computer_science',
    'conceptual_physics',
    'miscellaneous',
    'high_school_chemistry',
    'marketing',
    'professional_law',
    'management',
    'college_physics',
    'jurisprudence',
    'world_religions',
    'sociology',
    'us_foreign_policy',
    'high_school_macroeconomics',
    'computer_security',
    'moral_scenarios',
    'moral_disputes',
    'electrical_engineering',
    'astronomy',
    'college_biology',
]

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


@Benchmark.register(
    name='mmlu',
    pretty_name='MMLU',
    dataset_id='modelscope/mmlu',
    model_adapter=OutputType.GENERATION,
    output_types=[OutputType.MULTIPLE_CHOICE, OutputType.GENERATION],
    subset_list=SUBSET_LIST,
    metric_list=['AverageAccuracy'],
    few_shot_num=5,
    train_split='train',
    eval_split='test',
    prompt_template=
    """Answer the following multiple choice question about {subset_name}. The last line of your response should be of the following format: 'Answer: $LETTER' (without quotes) where LETTER is one of ABCD. Think step by step before answering.\n\n{query}""",  # noqa: E501
)
class MMLUAdapter(DataAdapter):

    def __init__(self, **kwargs):

        few_shot_num = kwargs.get('few_shot_num', 5)
        if few_shot_num > 5:
            logger.warning(f'few_shot_num <= 5 for MMLU, but got {few_shot_num}. Use 5-shot by default.')
            kwargs['few_shot_num'] = 5

        super().__init__(**kwargs)

        self.category_map = {k: v[-1] for k, v in SUBJECT_MAPPING.items()}
        self.choices = ['A', 'B', 'C', 'D']

    def load_from_disk(self, dataset_name_or_path, subset_list, work_dir, **kwargs) -> dict:
        data_dict = {}
        for subset_name in subset_list:
            data_dict[subset_name] = {}

            for split_name in [self.train_split, self.eval_split]:
                if split_name == 'train':
                    split_name_suffix = 'dev'
                elif split_name == 'test':
                    split_name_suffix = 'test'
                elif split_name == 'validation':
                    split_name_suffix = 'val'
                else:
                    raise ValueError(f'Invalid split name: {split_name}')

                if os.path.exists(dataset_name_or_path):
                    file_path = os.path.join(dataset_name_or_path, f'{subset_name}_{split_name_suffix}.csv')
                else:
                    file_path = os.path.join(work_dir, dataset_name_or_path, f'{subset_name}_{split_name_suffix}.csv')

                if os.path.exists(file_path):
                    with open(file_path, encoding='utf-8') as f:
                        rows = []
                        reader = csv.reader(f)
                        for row in reader:
                            if len(row) != 6:
                                logger.error(f'Mismatch len of row: {row}, len of row should be 6. Skip this row.')
                                continue
                            rows.append({
                                'input': row[0],
                                'A': row[1],
                                'B': row[2],
                                'C': row[3],
                                'D': row[4],
                                'target': row[5],
                            })

                        data_dict[subset_name].update({split_name: rows})

        return data_dict

    def gen_prompt(self, input_d: dict, subset_name: str, few_shot_list: list, **kwargs) -> dict:
        """
        Generate model prompt from raw input, unify the prompt format for MMLU benchmark.

        Args:
            input_d (dict): The raw input. A single data format of the MMLU:

            {'input': '___________ is based on the idea that customer expectations of the service they will receive shape their perception of the actual service encounter.',
            'A': 'Service quality.',
            'B': 'Service action.',
            'C': 'Service recovery.',
            'D': 'Service satisfaction.',
            'target': 'A'}

        Returns:
            {'data': [full_prompt], 'multi_choices': self.choices}

        """
        few_shot_prompts = [self._generate_prompt(input_d=sample, include_answer=True) for sample in few_shot_list]

        context: str = '\n'.join(few_shot_prompts) + '\n'
        context += self._generate_prompt(input_d=input_d, include_answer=False)

        full_prompt = self.prompt_template.format(subset_name=self._format_subject(subset_name), query=context.strip())

        return self.gen_prompt_data(full_prompt)

    def get_gold_answer(self, input_d: dict) -> str:
        # Get the gold choice
        return input_d.get('target', '')

    def parse_pred_result(self, result: str, raw_input_d: dict = None, eval_type: str = EvalType.CHECKPOINT) -> str:
        """
        Parse the model output to get the answer. Could be the best choice index.

        Args:
            result: Predicted answer from the model. Usually a string for chat.
            raw_input_d: The raw input. Depending on the dataset.
            eval_type: 'checkpoint' or 'service' or 'custom'

        Returns:
            The parsed answer. Depending on the dataset. Usually a string for chat.
        """
        if self.model_adapter == OutputType.MULTIPLE_CHOICE:
            return result
        else:
            return ResponseParser.parse_first_option(result, options=self.choices)

    def match(self, gold: str, pred: str) -> float:
        return exact_match(gold=gold, pred=pred)

    def _generate_prompt(self, input_d: dict, include_answer=True) -> str:

        input_choices: list = [input_d['A'], input_d['B'], input_d['C'], input_d['D']]

        example: str = input_d['input']
        for j in range(len(self.choices)):
            example += f'\n{self.choices[j]}) {input_choices[j]}'

        if include_answer:
            example += f"\nAnswer: {input_d['target']}\n\n"
        else:
            example += '\nAnswer: \n\n'

        return example

    @classmethod
    def _format_subject(cls, subject):
        l = subject.split('_')
        s = ''
        for entry in l:
            s += ' ' + entry
        return s
