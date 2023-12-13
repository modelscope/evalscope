# Copyright (c) Alibaba, Inc. and its affiliates.

from llmuses.benchmarks.data_adapter import DataAdapter
from llmuses.metrics.metrics import exact_match, weighted_mean
from llmuses.utils.logger import get_logger
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


SUBJECT_MAPPING = {'abstract_algebra': ['Abstract Algebra', 'math', 'STEM'],
                   'anatomy': ['Anatomy', 'health', 'other (business, health, misc.)'],
                   'astronomy': ['Astronomy', 'physics', 'STEM'],
                   'business_ethics': ['Business Ethics', 'business', 'other (business, health, misc.)'],
                   'clinical_knowledge': ['Clinical Knowledge', 'health', 'other (business, health, misc.)'],
                   'college_biology': ['College Biology', 'biology', 'STEM'],
                   'college_chemistry': ['College Chemistry', 'chemistry', 'STEM'],
                   'college_computer_science': ['College Computer Science', 'computer science', 'STEM'],
                   'college_mathematics': ['College Mathematics', 'math', 'STEM'],
                   'college_medicine': ['College Medicine', 'health', 'other (business, health, misc.)'],
                   'college_physics': ['College Physics', 'physics', 'STEM'],
                   'computer_security': ['Computer Security', 'computer science', 'STEM'],
                   'conceptual_physics': ['Conceptual Physics', 'physics', 'STEM'],
                   'econometrics': ['Econometrics', 'economics', 'social sciences'],
                   'electrical_engineering': ['Electrical Engineering', 'engineering', 'STEM'],
                   'elementary_mathematics': ['Elementary Mathematics', 'math', 'STEM'],
                   'formal_logic': ['Formal Logic', 'philosophy', 'humanities'],
                   'global_facts': ['Global Facts', 'other', 'other (business, health, misc.)'],
                   'high_school_biology': ['High School Biology', 'biology', 'STEM'],
                   'high_school_chemistry': ['High School Chemistry', 'chemistry', 'STEM'],
                   'high_school_computer_science': ['High School Computer Science', 'computer science', 'STEM'],
                   'high_school_european_history': ['High School European History', 'history', 'humanities'],
                   'high_school_geography': ['High School Geography', 'geography', 'social sciences'],
                   'high_school_government_and_politics': ['High School Government And Politics', 'politics', 'social sciences'],
                   'high_school_macroeconomics': ['High School Macroeconomics', 'economics', 'social sciences'],
                   'high_school_mathematics': ['High School Mathematics', 'math', 'STEM'],
                   'high_school_microeconomics': ['High School Microeconomics', 'economics', 'social sciences'],
                   'high_school_physics': ['High School Physics', 'physics', 'STEM'],
                   'high_school_psychology': ['High School Psychology', 'psychology', 'social sciences'],
                   'high_school_statistics': ['High School Statistics', 'math', 'STEM'],
                   'high_school_us_history': ['High School Us History', 'history', 'humanities'],
                   'high_school_world_history': ['High School World History', 'history', 'humanities'],
                   'human_aging': ['Human Aging', 'health', 'other (business, health, misc.)'],
                   'human_sexuality': ['Human Sexuality', 'culture', 'social sciences'],
                   'international_law': ['International Law', 'law', 'humanities'],
                   'jurisprudence': ['Jurisprudence', 'law', 'humanities'],
                   'logical_fallacies': ['Logical Fallacies', 'philosophy', 'humanities'],
                   'machine_learning': ['Machine Learning', 'computer science', 'STEM'],
                   'management': ['Management', 'business', 'other (business, health, misc.)'],
                   'marketing': ['Marketing', 'business', 'other (business, health, misc.)'],
                   'medical_genetics': ['Medical Genetics', 'health', 'other (business, health, misc.)'],
                   'miscellaneous': ['Miscellaneous', 'other', 'other (business, health, misc.)'],
                   'moral_disputes': ['Moral Disputes', 'philosophy', 'humanities'],
                   'moral_scenarios': ['Moral Scenarios', 'philosophy', 'humanities'],
                   'nutrition': ['Nutrition', 'health', 'other (business, health, misc.)'],
                   'philosophy': ['Philosophy', 'philosophy', 'humanities'],
                   'prehistory': ['Prehistory', 'history', 'humanities'],
                   'professional_accounting': ['Professional Accounting', 'other', 'other (business, health, misc.)'],
                   'professional_law': ['Professional Law', 'law', 'humanities'],
                   'professional_medicine': ['Professional Medicine', 'health', 'other (business, health, misc.)'],
                   'professional_psychology': ['Professional Psychology', 'psychology', 'social sciences'],
                   'public_relations': ['Public Relations', 'politics', 'social sciences'],
                   'security_studies': ['Security Studies', 'politics', 'social sciences'],
                   'sociology': ['Sociology', 'culture', 'social sciences'],
                   'us_foreign_policy': ['Us Foreign Policy', 'politics', 'social sciences'],
                   'virology': ['Virology', 'health', 'other (business, health, misc.)'],
                   'world_religions': ['World Religions', 'philosophy', 'humanities'],
                   }


class MMLUAdapter(DataAdapter):

    choices = ['A', 'B', 'C', 'D']

    def __init__(self,
                 subset_list: list = None,
                 metric_list: list = None,
                 few_shot_num: int = 5,
                 train_split: str = 'train',
                 eval_split: str = 'test',
                 **kwargs):

        if subset_list is None:
            subset_list = SUBSET_LIST

        if metric_list is None:
            metric_list = [{'name': 'WeightedAverageAccuracy', 'object': weighted_mean}]

        super().__init__(subset_list=subset_list,
                         metric_list=metric_list,
                         few_shot_num=few_shot_num,
                         train_split=train_split,
                         eval_split=eval_split,
                         **kwargs)

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
            {'data': [(context, continuation), ...]}

        """
        prompt = 'The following are multiple choice questions (with answers) about {}.\n\n'.format(
            self._format_subject(subset_name)
        )
        few_shot_prompts = [self._generate_prompt(input_d=sample, include_answer=True) for sample in few_shot_list]

        context: str = '\n'.join(few_shot_prompts) + '\n'
        context += self._generate_prompt(input_d=input_d, include_answer=False)
        context = prompt + context

        full_prompt: str = context.strip() + self._generate_prompt(input_d=input_d, include_answer=False)

        return {'data': [full_prompt], 'multi_choices': self.choices}

    def get_gold_answer(self, input_d: dict) -> str:
        # Get the gold choice
        return input_d.get('target', '')

    def parse_pred_result(self, result: str, raw_input_d: dict = None) -> str:
        """
        Parse the model output to get the answer. Could be the best choice index.

        Args:
            result: Predicted answer from the model. Usually a string for chat.
            raw_input_d: The raw input. Depending on the dataset.

        Returns:
            The parsed answer. Depending on the dataset. Usually a string for chat.
        """
        return result

    def match(self, gold: str, pred: str) -> float:
        return exact_match(gold=gold, pred=pred)

    def compute_metric(self, review_res_list: list) -> float:
        """
        Compute evaluation result by specific metric.

        Args:
            review_res_list: review score list, e.g. [0, 1, 1, 0, ...]

        Returns:
            The metric score.
        """
        items = [(score, 1.0) for score in review_res_list]
        return weighted_mean(items)

    def gen_report(self, subset_score_map: dict) -> dict:
        """
        Generate report for the evaluation.

        Args:
            subset_score_map: The subset-score mapping. e.g. {subset_name: (score, num), ...}

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

        # Get domain-subject mapping
        subject_review_map = {}
        for subset_name, (subset_score, num) in subset_score_map.items():
            domain_name: str = SUBJECT_MAPPING.get(subset_name)[2]
            if domain_name in subject_review_map:
                subject_review_map[domain_name].append((subset_name, subset_score, num))
            else:
                subject_review_map[domain_name] = [(subset_name, subset_score, num)]

        # Get domain score
        category_list = []
        for domain_name, domain_res_list in subject_review_map.items():
            domain_weighted_avg_acc = sum([score * num for _, score, num in domain_res_list]) / \
                                      sum([num for _, _, num in domain_res_list])
            category_list.append({'name': domain_name,
                                  'score': domain_weighted_avg_acc,
                                  'subset': [{'name': subset_name, 'score': subset_score}
                                             for subset_name, subset_score, _ in domain_res_list]})

        # Get final dict of report
        res_map = dict(name='MMLU',
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
