# Copyright (c) Alibaba, Inc. and its affiliates.

from typing import Any, Dict

from evalscope.api.benchmark import BenchmarkMeta, MultiChoiceAdapter
from evalscope.api.dataset import Sample
from evalscope.api.registry import register_benchmark
from evalscope.constants import Tags
from evalscope.utils.logger import get_logger

logger = get_logger()

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

# Based on the prompt template for Chinese evaluation
USER_PROMPT_TEMPLATE = """以下是中国关于{subject}的单项选择题，请选出其中的正确答案。你的回答的最后一行应该是这样的格式："答案：[LETTER]"（不带引号），其中 [LETTER] 是 A、B、C、D 中的一个。

问题：{question}
选项：
{choices}
""".lstrip()  # noqa: E501

FEWSHOT_TEMPLATE = """以下是一些示例问题：

{fewshot}

""".lstrip()


@register_benchmark(
    BenchmarkMeta(
        name='ceval',
        pretty_name='C-Eval',
        tags=[Tags.KNOWLEDGE, Tags.MULTIPLE_CHOICE, Tags.CHINESE],
        description=
        'C-Eval is a benchmark designed to evaluate the performance of AI models on Chinese exams across various subjects, including STEM, social sciences, and humanities. It consists of multiple-choice questions that test knowledge and reasoning abilities in these areas.',  # noqa: E501
        dataset_id='evalscope/ceval',
        subset_list=list(SUBJECT_MAPPING.keys()),
        metric_list=['acc'],
        few_shot_num=5,
        train_split='dev',
        eval_split='val',
        prompt_template=USER_PROMPT_TEMPLATE,
        few_shot_prompt_template=FEWSHOT_TEMPLATE,
    )
)
class CEVALAdapter(MultiChoiceAdapter):

    def __init__(self, **kwargs):

        super().__init__(**kwargs)

        self.category_map = {k: v[-1] for k, v in SUBJECT_MAPPING.items()}

    def record_to_sample(self, record: Dict[str, Any]) -> Sample:
        # Build choices list from A, B, C, D fields
        choices = [record['A'], record['B'], record['C'], record['D']]
        subset = self.current_subset_name

        return Sample(
            input=record['question'],
            choices=choices,
            target=record['answer'],
            metadata={
                'id': record.get('id', ''),
                'explanation': record.get('explanation', ''),
                'subject': subset
            },
        )

    def sample_to_fewshot(self, sample: Sample) -> str:
        q_str = f"""问题：{sample.input}"""
        choices = sample.choices if sample.choices is not None else []
        opt_str_list = []
        for i, choice in enumerate(choices):
            opt_str_list.append(f"""{chr(65 + i)}. {choice}""")
        opt_str = '\n'.join(opt_str_list)
        opt_str = f"""选项：\n{opt_str}"""
        exp_str = f"""解析：{sample.metadata.get('explanation', '')}"""
        ans_str = f"""答案：{sample.target}"""
        final_str = '\n'.join([q_str, opt_str, exp_str, ans_str])

        return final_str

    def format_fewshot_template(self, fewshot, sample):
        fewshot_str = FEWSHOT_TEMPLATE.format(fewshot=fewshot)
        prompt_str = self.format_prompt_template(sample)
        return fewshot_str + '\n' + prompt_str

    def format_prompt_template(self, sample):
        subject_name = SUBJECT_MAPPING.get(sample.metadata['subject'])[1]
        choices = sample.choices if sample.choices is not None else []
        choices_str = '\n'.join([f'{chr(65 + i)}. {choice}' for i, choice in enumerate(choices)])

        return USER_PROMPT_TEMPLATE.format(subject=subject_name, question=sample.input, choices=choices_str)

    def extract_answer(self, prediction, task_state) -> str:
        """
        Extract the answer from the prediction based on the task state.

        Args:
            prediction (str): The model's prediction string
            task_state (dict): The current task state containing metadata

        Returns:
            str: The extracted answer from the prediction
        """
        import re

        # Use regex to find the answer in the format "答案：LETTER"
        match = re.search(r'答案：([A-D])', prediction)
        if match:
            return match.group(1)
        else:
            logger.warning(f'No valid answer found in prediction: {prediction}')
            return ''
