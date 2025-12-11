# Copyright (c) Alibaba, Inc. and its affiliates.

import os
from typing import Any, Dict

from evalscope.api.benchmark import BenchmarkMeta, MultiChoiceAdapter
from evalscope.api.dataset import Sample
from evalscope.api.registry import register_benchmark
from evalscope.constants import Tags
from evalscope.utils.logger import get_logger
from evalscope.utils.multi_choices import FEW_SHOT_TEMPLATE, MultipleChoiceTemplate

logger = get_logger()

SUBSET_MAPPING = {
    'Electronic Science and Technology': ['Engineering'],
    'Philosophy': ['Philosophy'],
    'Traditional Chinese Medicine': ['Medicine'],
    'Applied Economics': ['Economics'],
    'Mathematics': ['Science'],
    'Physics': ['Science'],
    'Clinical Medicine': ['Medicine'],
    'Computer Science and Technology': ['Engineering'],
    'Information and Communication Engineering': ['Engineering'],
    'Control Science and Engineering': ['Engineering'],
    'Theoretical Economics': ['Economics'],
    'Law': ['Law'],
    'History': ['History'],
    'Basic Medicine': ['Medicine'],
    'Education': ['Education'],
    'Materials Science and Engineering': ['Engineering'],
    'Electrical Engineering': ['Engineering'],
    'Systems Science': ['Science'],
    'Power Engineering and Engineering Thermophysics': ['Engineering'],
    'Military Science': ['Military Science'],
    'Biology': ['Science'],
    'Business Administration': ['Management'],
    'Language and Literature': ['Literature and Arts'],
    'Public Health and Preventive Medicine': ['Medicine'],
    'Political Science': ['Law'],
    'Chemistry': ['Science'],
    'Hydraulic Engineering': ['Engineering'],
    'Chemical Engineering and Technology': ['Engineering'],
    'Pharmacy': ['Medicine'],
    'Geography': ['Science'],
    'Art Studies': ['Literature and Arts'],
    'Architecture': ['Engineering'],
    'Forestry Engineering': ['Engineering'],
    'Public Administration': ['Management'],
    'Oceanography': ['Science'],
    'Journalism and Communication': ['Literature and Arts'],
    'Nuclear Science and Technology': ['Engineering'],
    'Weapon Science and Technology': ['Engineering'],
    'Naval Architecture and Ocean Engineering': ['Engineering'],
    'Environmental Science and Engineering': ['Engineering'],
    'Transportation Engineering': ['Engineering'],
    'Geology': ['Science'],
    'Physical Oceanography': ['Science'],
    'Musicology': ['Literature and Arts'],
    'Stomatology': ['Medicine'],
    'Aquaculture': ['Agronomy'],
    'Mechanical Engineering': ['Engineering'],
    'Aeronautical and Astronautical Science and Technology': ['Engineering'],
    'Civil Engineering': ['Engineering'],
    'Mechanics': ['Engineering'],
    'Petroleum and Natural Gas Engineering': ['Engineering'],
    'Sociology': ['Sociology'],
    'Food Science and Engineering': ['Engineering'],
    'Agricultural Engineering': ['Engineering'],
    'Surveying and Mapping Science and Technology': ['Engineering'],
    'Metallurgical Engineering': ['Engineering'],
    'Library, Information and Archival Management': ['Management'],
    'Mining Engineering': ['Engineering'],
    'Astronomy': ['Science'],
    'Geological Resources and Geological Engineering': ['Engineering'],
    'Atmospheric Science': ['Science'],
    'Optical Engineering': ['Engineering'],
    'Animal Husbandry': ['Agronomy'],
    'Geophysics': ['Science'],
    'Crop Science': ['Agronomy'],
    'Management Science and Engineering': ['Management'],
    'Psychology': ['Education'],
    'Forestry': ['Agronomy'],
    'Textile Science and Engineering': ['Engineering'],
    'Veterinary Medicine': ['Agronomy'],
    'Instrument Science and Technology': ['Engineering'],
    'Physical Education': ['Education']
}


@register_benchmark(
    BenchmarkMeta(
        name='super_gpqa',
        pretty_name='SuperGPQA',
        tags=[Tags.KNOWLEDGE, Tags.MULTIPLE_CHOICE],
        description=
        'SuperGPQA is a large-scale multiple-choice question answering dataset, designed to evaluate the generalization ability of models across different fields. It contains 26,000+ questions from 50+ fields, with each question having 10 options.',  # noqa: E501
        dataset_id='m-a-p/SuperGPQA',
        subset_list=list(SUBSET_MAPPING.keys()),
        metric_list=['acc'],
        few_shot_num=0,
        train_split=None,
        eval_split='train',  # only have train split
        prompt_template=MultipleChoiceTemplate.SINGLE_ANSWER_COT,
    )
)
class SuperGPQAAdapter(MultiChoiceAdapter):

    def __init__(self, **kwargs):

        super().__init__(**kwargs)
        if self.few_shot_num > 0 and self.few_shot_num != 5:
            logger.warning(
                f'Only support few_shot_num 0 or 5 for SuperGPQA, but got {self.few_shot_num}. Use 5-shot by default.'
            )
            self.few_shot_num = 5

        self.reformat_subset = True
        self.category_map = SUBSET_MAPPING

    def record_to_sample(self, record: Dict[str, Any]) -> Sample:
        return Sample(
            input=record['question'],
            choices=record['options'],
            target=record['answer_letter'],
            subset_key=record['field'],
            metadata={
                'field': record['field'],
                'discipline': record['discipline'],
                'uuid': record.get('uuid', ''),
                'explanation': record.get('answer', ''),
            },
        )

    def format_fewshot_template(self, fewshot, sample):
        from .prompt import FEW_SHOT_SAMPLES

        return FEW_SHOT_TEMPLATE.format(fewshot=FEW_SHOT_SAMPLES, ) + self.format_prompt_template(sample)

    def extract_answer(self, prediction: str, task_state) -> str:
        """
        Extract the answer from the prediction.
        """
        from .utils import extract_option_content, extract_option_labels

        choices = [choice.value for choice in task_state.choices]
        if self.few_shot_num == 0:
            predict = extract_option_labels(prediction, 'ABCDEFGHIJ')
            if predict is None:
                # Try to extract by content matching
                predict = extract_option_content(prediction, choices)
                predict = chr(choices.index(predict) + 65) if predict else None
        else:
            response = prediction.split('Question:')[0]
            predict = extract_option_labels(response, 'ABCDEFGHIJ')
            if predict is None:
                predict = extract_option_content(response, choices)
                predict = chr(choices.index(predict) + 65) if predict else None
            if predict is None:
                predict = extract_option_labels(prediction, 'ABCDEFGHIJ')
                if predict is None:
                    predict = extract_option_content(prediction, choices)
                    predict = chr(choices.index(predict) + 65) if predict else None

        return predict or ''
