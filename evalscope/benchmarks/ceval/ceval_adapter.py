# Copyright (c) Alibaba, Inc. and its affiliates.
import csv
import os

from evalscope.benchmarks import Benchmark, DataAdapter
from evalscope.constants import EvalType
from evalscope.metrics import WeightedAverageAccuracy
from evalscope.metrics.metrics import exact_match, weighted_mean
from evalscope.models import MultiChoiceModelAdapter
from evalscope.utils import ResponseParser, normalize_score
from evalscope.utils.logger import get_logger

# flake8: noqa

logger = get_logger()

SUBSET_LIST = [
    'computer_network',
    'operating_system',
    'computer_architecture',
    'college_programming',
    'college_physics',
    'college_chemistry',
    'advanced_mathematics',
    'probability_and_statistics',
    'discrete_mathematics',
    'electrical_engineer',
    'metrology_engineer',
    'high_school_mathematics',
    'high_school_physics',
    'high_school_chemistry',
    'high_school_biology',
    'middle_school_mathematics',
    'middle_school_biology',
    'middle_school_physics',
    'middle_school_chemistry',
    'veterinary_medicine',
    'college_economics',
    'business_administration',
    'marxism',
    'mao_zedong_thought',
    'education_science',
    'teacher_qualification',
    'high_school_politics',
    'high_school_geography',
    'middle_school_politics',
    'middle_school_geography',
    'modern_chinese_history',
    'ideological_and_moral_cultivation',
    'logic',
    'law',
    'chinese_language_and_literature',
    'art_studies',
    'professional_tour_guide',
    'legal_professional',
    'high_school_chinese',
    'high_school_history',
    'middle_school_history',
    'civil_servant',
    'sports_science',
    'plant_protection',
    'basic_medicine',
    'clinical_medicine',
    'urban_and_rural_planner',
    'accountant',
    'fire_engineer',
    'environmental_impact_assessment_engineer',
    'tax_accountant',
    'physician',
]

SUBJECT_MAPPING = {
    'computer_network': ['Computer Network', '计算机网络', 'STEM'],
    'operating_system': ['Operating System', '操作系统', 'STEM'],
    'computer_architecture': ['Computer Architecture', '计算机组成', 'STEM'],
    'college_programming': ['College Programming', '大学编程', 'STEM'],
    'college_physics': ['College Physics', '大学物理', 'STEM'],
    'college_chemistry': ['College Chemistry', '大学化学', 'STEM'],
    'advanced_mathematics': ['Advanced Mathematics', '高等数学', 'STEM'],
    'probability_and_statistics': ['Probability and Statistics', '概率统计', 'STEM'],
    'discrete_mathematics': ['Discrete Mathematics', '离散数学', 'STEM'],
    'electrical_engineer': ['Electrical Engineer', '注册电气工程师', 'STEM'],
    'metrology_engineer': ['Metrology Engineer', '注册计量师', 'STEM'],
    'high_school_mathematics': ['High School Mathematics', '高中数学', 'STEM'],
    'high_school_physics': ['High School Physics', '高中物理', 'STEM'],
    'high_school_chemistry': ['High School Chemistry', '高中化学', 'STEM'],
    'high_school_biology': ['High School Biology', '高中生物', 'STEM'],
    'middle_school_mathematics': ['Middle School Mathematics', '初中数学', 'STEM'],
    'middle_school_biology': ['Middle School Biology', '初中生物', 'STEM'],
    'middle_school_physics': ['Middle School Physics', '初中物理', 'STEM'],
    'middle_school_chemistry': ['Middle School Chemistry', '初中化学', 'STEM'],
    'veterinary_medicine': ['Veterinary Medicine', '兽医学', 'STEM'],
    'college_economics': ['College Economics', '大学经济学', 'Social Science'],
    'business_administration': ['Business Administration', '工商管理', 'Social Science'],
    'marxism': ['Marxism', '马克思主义基本原理', 'Social Science'],
    'mao_zedong_thought': ['Mao Zedong Thought', '毛泽东思想和中国特色社会主义理论体系概论', 'Social Science'],
    'education_science': ['Education Science', '教育学', 'Social Science'],
    'teacher_qualification': ['Teacher Qualification', '教师资格', 'Social Science'],
    'high_school_politics': ['High School Politics', '高中政治', 'Social Science'],
    'high_school_geography': ['High School Geography', '高中地理', 'Social Science'],
    'middle_school_politics': ['Middle School Politics', '初中政治', 'Social Science'],
    'middle_school_geography': ['Middle School Geography', '初中地理', 'Social Science'],
    'modern_chinese_history': ['Modern Chinese History', '近代史纲要', 'Humanities'],
    'ideological_and_moral_cultivation': ['Ideological and Moral Cultivation', '思想道德修养与法律基础', 'Humanities'],
    'logic': ['Logic', '逻辑学', 'Humanities'],
    'law': ['Law', '法学', 'Humanities'],
    'chinese_language_and_literature': ['Chinese Language and Literature', '中国语言文学', 'Humanities'],
    'art_studies': ['Art Studies', '艺术学', 'Humanities'],
    'professional_tour_guide': ['Professional Tour Guide', '导游资格', 'Humanities'],
    'legal_professional': ['Legal Professional', '法律职业资格', 'Humanities'],
    'high_school_chinese': ['High School Chinese', '高中语文', 'Humanities'],
    'high_school_history': ['High School History', '高中历史', 'Humanities'],
    'middle_school_history': ['Middle School History', '初中历史', 'Humanities'],
    'civil_servant': ['Civil Servant', '公务员', 'Other'],
    'sports_science': ['Sports Science', '体育学', 'Other'],
    'plant_protection': ['Plant Protection', '植物保护', 'Other'],
    'basic_medicine': ['Basic Medicine', '基础医学', 'Other'],
    'clinical_medicine': ['Clinical Medicine', '临床医学', 'Other'],
    'urban_and_rural_planner': ['Urban and Rural Planner', '注册城乡规划师', 'Other'],
    'accountant': ['Accountant', '注册会计师', 'Other'],
    'fire_engineer': ['Fire Engineer', '注册消防工程师', 'Other'],
    'environmental_impact_assessment_engineer': ['Environmental Impact Assessment Engineer', '环境影响评价工程师', 'Other'],
    'tax_accountant': ['Tax Accountant', '税务师', 'Other'],
    'physician': ['Physician', '医师资格', 'Other']
}


@Benchmark.register(
    name='ceval',
    dataset_id='modelscope/ceval-exam',
    model_adapter=MultiChoiceModelAdapter,
    subset_list=SUBSET_LIST,
    metric_list=[WeightedAverageAccuracy],
    few_shot_num=0,
    train_split='dev',
    eval_split='val',
)
class CEVALAdapter(DataAdapter):

    choices = ['A', 'B', 'C', 'D']

    def __init__(self, **kwargs):

        few_shot_num = kwargs.get('few_shot_num', 0)
        if few_shot_num > 5:
            logger.warning(f'few_shot_num <= 5 for C-Eval, but got {few_shot_num}. Use 5-shot by default.')
            kwargs['few_shot_num'] = 5

        super().__init__(**kwargs)

    def load_from_disk(self, dataset_name_or_path, subset_list, work_dir, **kwargs) -> dict:
        data_dict = {}
        for subset_name in subset_list:
            for split_name in [self.train_split, self.eval_split]:
                if os.path.exists(dataset_name_or_path):
                    file_path = os.path.join(dataset_name_or_path, f'{subset_name}_{split_name}.csv')
                else:
                    file_path = os.path.join(work_dir, dataset_name_or_path, f'{subset_name}_{split_name}.csv')
                if os.path.exists(file_path):
                    with open(file_path, encoding='utf-8') as f:
                        rows = []
                        reader = csv.reader(f)
                        header = next(reader)
                        for row in reader:
                            item = dict(zip(header, row))
                            item.setdefault('explanation', '')
                            item.setdefault('answer', '')
                            rows.append(item)

                        if subset_name in data_dict:
                            data_dict[subset_name].update({split_name: rows})
                        else:
                            data_dict[subset_name] = {split_name: rows}

        return data_dict

    def gen_prompt(self, input_d: dict, subset_name: str, few_shot_list: list, **kwargs) -> dict:
        """
        Generate model prompt from raw input, unify the prompt format for C-Eval benchmark.

        Args:
            input_d (dict): The raw input. A single data format of the C-Eval:

            {'id': 0,
            'question': '下列关于税法基本原则的表述中，不正确的是____。',
            'A': '税收法定原则包括税收要件法定原则和税务合法性原则',
            'B': '税收公平原则源于法律上的平等性原则',
            'C': '税收效率原则包含经济效率和行政效率两个方面',
            'D': '税务机关按法定程序依法征税，可以自由做出减征、停征或免征税款的决定',
            'answer': 'D',
            'explanation': ''}

        Returns:
            {'data': ['prompt ...']}
        """

        few_shot_prompts = [self._format_example(input_d=sample, include_answer=True) for sample in few_shot_list]

        if len(few_shot_prompts) > 0:
            context: str = '\n'.join(few_shot_prompts) + '\n'
        else:
            context = ''

        full_prompt: str = context.strip() + self._format_example(input_d=input_d, include_answer=False)

        subject_name: str = SUBJECT_MAPPING.get(subset_name)[1] if SUBJECT_MAPPING.get(subset_name) else subset_name
        full_prompt = f'以下是中国关于{subject_name}考试的单项选择题，请选出其中的正确答案。\n' + full_prompt

        return {'data': [full_prompt], 'multi_choices': self.choices}

    def get_gold_answer(self, input_d: dict) -> str:
        # Get the gold choice
        return input_d.get('answer', '')

    def parse_pred_result(self, result: str, raw_input_d: dict = None, eval_type: str = EvalType.CHECKPOINT) -> str:
        """
        Parse the model output to get the answer. Could be the best choice index.

        Args:
            result: Predicted answer from the model. Usually a string for chat.
            raw_input_d (dict): The raw input. Depending on the dataset.
            eval_type: `checkpoint` or `service` or `custom`. Default is `checkpoint`.

        Returns:
            The parsed answer. Depending on the dataset. Usually a string for chat.
        """
        if eval_type == EvalType.CHECKPOINT:
            return result
        elif eval_type == EvalType.SERVICE:
            return ResponseParser.parse_first_option_with_choices(result, self.choices)  # TODO: to be checked !
        elif eval_type == EvalType.CUSTOM:
            return ResponseParser.parse_first_option_with_choices(result, self.choices)  # TODO: to be checked !
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
            "name":"C-Eval",
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
            domain_name: str = SUBJECT_MAPPING.get(subset_name)[2] if SUBJECT_MAPPING.get(subset_name) else 'DEFAULT'
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
            name=report_name or 'ceval',
            metric=self.metric_list[0]['name'],
            score=weighted_avg_acc,
            category=category_list,
            total_num=total_num)

        return res_map

    @classmethod
    def _format_example(cls, input_d: dict, include_answer=True):
        example = '问题：' + input_d['question']
        for choice in cls.choices:
            example += f'\n{choice}. {input_d[f"{choice}"]}'

        if include_answer:
            example += '\n答案: ' + input_d['answer'] + '\n\n'
        else:
            example += '\n答案: '
        return example
