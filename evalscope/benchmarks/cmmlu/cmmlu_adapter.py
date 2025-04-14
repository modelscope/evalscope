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
    'agronomy', 'anatomy', 'ancient_chinese', 'arts', 'astronomy', 'business_ethics', 'chinese_civil_service_exam',
    'chinese_driving_rule', 'chinese_food_culture', 'chinese_foreign_policy', 'chinese_history', 'chinese_literature',
    'chinese_teacher_qualification', 'college_actuarial_science', 'college_education', 'college_engineering_hydrology',
    'college_law', 'college_mathematics', 'college_medical_statistics', 'clinical_knowledge', 'college_medicine',
    'computer_science', 'computer_security', 'conceptual_physics', 'construction_project_management', 'economics',
    'education', 'elementary_chinese', 'elementary_commonsense', 'elementary_information_and_technology',
    'electrical_engineering', 'elementary_mathematics', 'ethnology', 'food_science', 'genetics', 'global_facts',
    'high_school_biology', 'high_school_chemistry', 'high_school_geography', 'high_school_mathematics',
    'high_school_physics', 'high_school_politics', 'human_sexuality', 'international_law', 'journalism',
    'jurisprudence', 'legal_and_moral_basis', 'logical', 'machine_learning', 'management', 'marketing',
    'marxist_theory', 'modern_chinese', 'nutrition', 'philosophy', 'professional_accounting', 'professional_law',
    'professional_medicine', 'professional_psychology', 'public_relations', 'security_study', 'sociology',
    'sports_science', 'traditional_chinese_medicine', 'virology', 'world_history', 'world_religions'
]

SUBJECT_MAPPING = {
    'agronomy': ['other', 'Other'],
    'anatomy': ['biology', 'STEM'],
    'ancient_chinese': ['china specific', 'China specific'],
    'arts': ['arts', 'Humanities'],
    'astronomy': ['physics', 'STEM'],
    'business_ethics': ['business', 'Social Science'],
    'chinese_civil_service_exam': ['china specific', 'China specific'],
    'chinese_driving_rule': ['china specific', 'China specific'],
    'chinese_food_culture': ['china specific', 'China specific'],
    'chinese_foreign_policy': ['china specific', 'China specific'],
    'chinese_history': ['china specific', 'China specific'],
    'chinese_literature': ['china specific', 'China specific'],
    'chinese_teacher_qualification': ['china specific', 'China specific'],
    'college_actuarial_science': ['math', 'STEM'],
    'college_education': ['education', 'Social Science'],
    'college_engineering_hydrology': ['engineering', 'STEM'],
    'college_law': ['law', 'Humanities'],
    'college_mathematics': ['math', 'STEM'],
    'college_medical_statistics': ['statistics', 'STEM'],
    'clinical_knowledge': ['other', 'Other'],
    'college_medicine': ['other', 'Other'],
    'computer_science': ['computer science', 'STEM'],
    'computer_security': ['other', 'Other'],
    'conceptual_physics': ['physics', 'STEM'],
    'construction_project_management': ['china specific', 'China specific'],
    'economics': ['economics', 'Social Science'],
    'education': ['education', 'Social Science'],
    'elementary_chinese': ['china specific', 'China specific'],
    'elementary_commonsense': ['china specific', 'China specific'],
    'elementary_information_and_technology': ['other', 'Other'],
    'electrical_engineering': ['engineering', 'STEM'],
    'elementary_mathematics': ['math', 'STEM'],
    'ethnology': ['china specific', 'China specific'],
    'food_science': ['other', 'Other'],
    'genetics': ['biology', 'STEM'],
    'global_facts': ['global', 'Humanities'],
    'high_school_biology': ['biology', 'STEM'],
    'high_school_chemistry': ['chemistry', 'STEM'],
    'high_school_geography': ['geography', 'Social Science'],
    'high_school_mathematics': ['math', 'STEM'],
    'high_school_physics': ['physics', 'STEM'],
    'high_school_politics': ['china specific', 'China specific'],
    'human_sexuality': ['other', 'Other'],
    'international_law': ['law', 'Humanities'],
    'journalism': ['sociology', 'Social Science'],
    'jurisprudence': ['law', 'Humanities'],
    'legal_and_moral_basis': ['other', 'Other'],
    'logical': ['philosophy', 'Humanities'],
    'machine_learning': ['computer science', 'STEM'],
    'management': ['business', 'Social Science'],
    'marketing': ['business', 'Social Science'],
    'marxist_theory': ['philosophy', 'Humanities'],
    'modern_chinese': ['china specific', 'China specific'],
    'nutrition': ['other', 'Other'],
    'philosophy': ['philosophy', 'Humanities'],
    'professional_accounting': ['business', 'Social Science'],
    'professional_law': ['law', 'Humanities'],
    'professional_medicine': ['other', 'Other'],
    'professional_psychology': ['psychology', 'Social Science'],
    'public_relations': ['politics', 'Social Science'],
    'security_study': ['politics', 'Social Science'],
    'sociology': ['culture', 'Social Science'],
    'sports_science': ['other', 'Other'],
    'traditional_chinese_medicine': ['china specific', 'China specific'],
    'virology': ['biology', 'STEM'],
    'world_history': ['history', 'Humanities'],
    'world_religions': ['global', 'Humanities']
}


@Benchmark.register(
    name='cmmlu',
    pretty_name='C-MMLU',
    dataset_id='modelscope/cmmlu',
    model_adapter=OutputType.GENERATION,
    output_types=[OutputType.MULTIPLE_CHOICE, OutputType.GENERATION],
    subset_list=SUBSET_LIST,
    metric_list=['AverageAccuracy'],
    few_shot_num=5,
    train_split='dev',
    eval_split='test',
    prompt_template=
    '以下是关于{subset_name}的单项选择题，请给出正确答案的选项。你的回答的最后一行应该是这样的格式：“答案：LETTER”（不带引号），其中 LETTER 是 A、B、C、D 中的一个。\n{query}',
)
class CMMLUAdapter(DataAdapter):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.category_map = {k: v[-1] for k, v in SUBJECT_MAPPING.items()}
        self.choices = ['A', 'B', 'C', 'D']

    def load_from_disk(self, dataset_name_or_path, subset_list, work_dir, **kwargs) -> dict:
        data_dict = {}
        for subset_name in subset_list:
            data_dict[subset_name] = {}
            for split_name in [self.train_split, self.eval_split]:
                file_path = os.path.join(work_dir, dataset_name_or_path, split_name, f'{subset_name}.csv')
                if os.path.exists(file_path):
                    with open(file_path, encoding='utf-8') as f:
                        rows = []
                        reader = csv.reader(f)
                        for row in reader:
                            if len(row) != 7:
                                logger.error(f'Mismatch len of row: {row}, len of row should be 6. Skip this row.')
                                continue
                            rows.append({
                                'Question': row[1],
                                'A': row[2],
                                'B': row[3],
                                'C': row[4],
                                'D': row[5],
                                'Answer': row[6],
                            })

                        data_dict[subset_name].update({split_name: rows})

        return data_dict

    def gen_prompt(self, input_d: dict, subset_name: str, few_shot_list: list, **kwargs) -> dict:
        """
        Generate model prompt from raw input, unify the prompt format for CMMLU benchmark.

        Args:
            input_d (dict): The raw input. A single data format of the CMMLU:

            {'Question': '下列关于重力的说法正确的是',
            'A': '在地球周围的物体都要受到重力作用，与其运动状态无关',
            'B': '对某一物体而言，重力的大小是一个恒量，不随物体的地理位置而改变',
            'C': '重力就是地球对物体的吸引力，重力的方向总是竖直向下',
            'D': '在地球表面各处的重力方向都是相同的',
            'Answer': 'A'}

        Returns:
            {'data': [(context, continuation), ...]}

        """
        few_shot_prompts = [self._generate_prompt(input_d=sample, include_answer=True) for sample in few_shot_list]
        context = '\n'.join(few_shot_prompts) + '\n'
        context += self._generate_prompt(input_d=input_d, include_answer=False)

        full_prompt = self.prompt_template.format(subset_name=self._format_subject(subset_name), query=context.strip())

        return self.gen_prompt_data(full_prompt)

    def get_gold_answer(self, input_d: dict) -> str:
        # Get the gold choice
        return input_d.get('Answer', '')

    def parse_pred_result(self, result: str, raw_input_d: dict = None, eval_type: str = EvalType.CHECKPOINT) -> str:
        """
        Parse the model output to get the answer. Could be the best choice index.

        Args:
            result: Predicted answer from the model. Usually a string for chat.
            raw_input_d: The raw input. Depending on the dataset.
            eval_type: The evaluation type. 'checkpoint', 'service', 'custom'.

        Returns:
            The parsed answer. Depending on the dataset. Usually a string for chat.
        """
        if self.model_adapter == OutputType.MULTIPLE_CHOICE:
            return result
        else:
            return ResponseParser.parse_first_option_with_choices(text=result, options=self.choices)

    def match(self, gold: str, pred: str) -> float:
        return exact_match(gold=gold, pred=pred)

    def _generate_prompt(self, input_d: dict, include_answer=True) -> str:

        input_choices: list = [input_d['A'], input_d['B'], input_d['C'], input_d['D']]

        example: str = input_d['Question']
        for j in range(len(self.choices)):
            example += '\n{}. {}'.format(self.choices[j], input_choices[j])

        example += '\nAnswer:'
        if include_answer:
            example += ' {}\n\n'.format(input_d['Answer'])

        return example

    @classmethod
    def _format_subject(cls, subject):
        l = subject.split('_')
        s = ''
        for entry in l:
            s += ' ' + entry
        return s
