# Copyright (c) Alibaba, Inc. and its affiliates.
import csv
import os

from evalscope.benchmarks import Benchmark, DataAdapter
from evalscope.constants import EvalType
from evalscope.metrics import WeightedAverageAccuracy, exact_match
from evalscope.models import MultiChoiceModelAdapter
from evalscope.utils import ResponseParser, normalize_score
from evalscope.utils.logger import get_logger

# flake8: noqa

logger = get_logger()

DATASET_ID = 'modelscope/mmlu'

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
    dataset_id='modelscope/mmlu',
    model_adapter=MultiChoiceModelAdapter,
    subset_list=SUBSET_LIST,
    metric_list=[WeightedAverageAccuracy],
    few_shot_num=5,
    train_split='train',
    eval_split='test',
    prompt_template='',
)
class MMLUAdapter(DataAdapter):

    choices = ['A', 'B', 'C', 'D']

    def __init__(self, **kwargs):

        few_shot_num = kwargs.get('few_shot_num', 5)
        if few_shot_num > 5:
            logger.warning(f'few_shot_num <= 5 for MMLU, but got {few_shot_num}. Use 5-shot by default.')
            kwargs['few_shot_num'] = 5

        super().__init__(**kwargs)

    def load_from_disk(self, dataset_name_or_path, subset_list, work_dir, **kwargs) -> dict:
        data_dict = {}
        for subset_name in subset_list:
            data_dict[subset_name] = {}

            for split_name in [self.train_split, self.eval_split]:
                if self.train_split == 'train':
                    split_name_suffix = 'dev'
                elif self.eval_split == 'test':
                    split_name_suffix = 'test'
                elif self.eval_split == 'validation':
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
        prompt = 'The following are multiple choice questions (with answers) about {}.\n\n'.format(
            self._format_subject(subset_name))
        few_shot_prompts = [self._generate_prompt(input_d=sample, include_answer=True) for sample in few_shot_list]

        context: str = '\n'.join(few_shot_prompts) + '\n'
        context += self._generate_prompt(input_d=input_d, include_answer=False)
        context = prompt + context

        full_prompt: str = context.strip() + self._generate_prompt(input_d=input_d, include_answer=False)

        return {'data': [full_prompt], 'multi_choices': self.choices}

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
        if eval_type == EvalType.CHECKPOINT:
            return result
        elif eval_type == EvalType.SERVICE:
            return ResponseParser.parse_first_option_with_choices(result, self.choices)
        elif eval_type == EvalType.CUSTOM:
            return ResponseParser.parse_first_option_with_choices(result, self.choices)
        else:
            raise ValueError(f'Invalid eval_type: {eval_type}')

    def match(self, gold: str, pred: str) -> float:
        return exact_match(gold=gold, pred=pred)

    def gen_report(self, subset_score_map: dict, report_name: str = None) -> dict:
        """
        Generate report for the evaluation.

        Args:
            subset_score_map: The subset-score mapping. e.g. {subset_name: (score, num), ...}
            report_name: The user-defined report name.

        Returns:
        {
            "name":"MMLU",
            "metric":"WeightedAverageAccuracy",
            "score":0.3389,
            "category":[
                {
                    "name":"STEM",
                    "score":0.2528,
                    "subset":[
                        {
                            "name":"computer_network",
                            "score":0.2632
                        },
                        {
                            "name":"operating_system",
                            "score":0.3157
                        },
                        {
                            "name":"computer_architecture",
                            "score":0.4285
                        }
                    ]
                }
            ],
            "total_num":59
        }
        """
        total_num: int = sum([num for _, num in subset_score_map.values()])
        weighted_avg_acc: float = sum([score * num for score, num in subset_score_map.values()]) / total_num
        weighted_avg_acc = normalize_score(score=weighted_avg_acc)

        # Get domain-subject mapping
        subject_review_map = {}
        for subset_name, (subset_score, num) in subset_score_map.items():
            domain_name: str = SUBJECT_MAPPING.get(subset_name)[2] if SUBJECT_MAPPING.get(subset_name) else subset_name
            if domain_name in subject_review_map:
                subject_review_map[domain_name].append((subset_name, subset_score, num))
            else:
                subject_review_map[domain_name] = [(subset_name, subset_score, num)]

        # Get domain score
        category_list = []
        for domain_name, domain_res_list in subject_review_map.items():
            domain_weighted_avg_acc = sum([score * num for _, score, num in domain_res_list]) / \
                                      sum([num for _, _, num in domain_res_list])
            domain_weighted_avg_acc = normalize_score(score=domain_weighted_avg_acc)
            category_list.append({
                'name':
                domain_name,
                'score':
                domain_weighted_avg_acc,
                'subset': [{
                    'name': subset_name,
                    'score': normalize_score(score=subset_score)
                } for subset_name, subset_score, _ in domain_res_list]
            })

        category_list = sorted(category_list, key=lambda x: x['name'])

        # Get final dict of report
        res_map = dict(
            name=report_name or 'mmlu',
            metric=self.metric_list[0]['name'],
            score=weighted_avg_acc,
            category=category_list,
            total_num=total_num)

        return res_map

    @classmethod
    def _generate_prompt(cls, input_d: dict, include_answer=True) -> str:

        input_choices: list = [input_d['A'], input_d['B'], input_d['C'], input_d['D']]

        example: str = input_d['input']
        for j in range(len(cls.choices)):
            example += '\n{}. {}'.format(cls.choices[j], input_choices[j])

        example += '\nAnswer:'
        if include_answer:
            example += ' {}\n\n'.format(input_d['target'])

        return example

    @classmethod
    def _format_subject(cls, subject):
        l = subject.split('_')
        s = ''
        for entry in l:
            s += ' ' + entry
        return s
