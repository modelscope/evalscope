# Copyright (c) Alibaba, Inc. and its affiliates.

from evalscope.api.benchmark import BenchmarkMeta, MultiChoiceAdapter
from evalscope.api.dataset import Sample
from evalscope.api.registry import register_benchmark
from evalscope.constants import Tags
from evalscope.utils.logger import get_logger
from evalscope.utils.multi_choices import MultipleChoiceTemplate

# flake8: noqa

logger = get_logger()

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


@register_benchmark(
    BenchmarkMeta(
        name='cmmlu',
        pretty_name='C-MMLU',
        tags=[Tags.KNOWLEDGE, Tags.MULTIPLE_CHOICE, Tags.CHINESE],
        description=
        'C-MMLU is a benchmark designed to evaluate the performance of AI models on Chinese language tasks, including reading comprehension, text classification, and more.',
        dataset_id='evalscope/cmmlu',
        metric_list=['acc'],
        subset_list=list(SUBJECT_MAPPING.keys()),
        few_shot_num=0,
        train_split=None,
        eval_split='test',
        prompt_template=MultipleChoiceTemplate.CHINESE_SINGLE_ANSWER_TEMPLATE_COT,
    )
)
class CMMLUAdapter(MultiChoiceAdapter):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.reformat_subset = True
        self.category_map = {k: v[-1] for k, v in SUBJECT_MAPPING.items()}

    def record_to_sample(self, record) -> Sample:

        # choices: ["(A) 农业生产工具","(B) 土地","(C) 劳动力","(D) 资金"]
        # remove the leading (A), (B), (C), (D)
        raw_choices = record['choices']
        choice_list = [choice[3:].strip() for choice in raw_choices]

        return Sample(
            input=record['question'],
            choices=choice_list,
            target=record['answer'][1],  # answer is like "A"
            subset_key=record['category'],
            metadata={'subject': record['category']},
        )
