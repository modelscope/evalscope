import os
import random
import re

from evalscope.benchmarks import Benchmark, DataAdapter
from evalscope.constants import EvalType, OutputType
from evalscope.metrics import exact_match
from evalscope.utils import logger

current_dir = os.path.dirname(os.path.abspath(__file__))

SUBSET_LIST = [
    'Electronic Science and Technology', 'Philosophy', 'Traditional Chinese Medicine', 'Applied Economics',
    'Mathematics', 'Physics', 'Clinical Medicine', 'Computer Science and Technology',
    'Information and Communication Engineering', 'Control Science and Engineering', 'Theoretical Economics', 'Law',
    'History', 'Basic Medicine', 'Education', 'Materials Science and Engineering', 'Electrical Engineering',
    'Systems Science', 'Power Engineering and Engineering Thermophysics', 'Military Science', 'Biology',
    'Business Administration', 'Language and Literature', 'Public Health and Preventive Medicine', 'Political Science',
    'Chemistry', 'Hydraulic Engineering', 'Chemical Engineering and Technology', 'Pharmacy', 'Geography', 'Art Studies',
    'Architecture', 'Forestry Engineering', 'Public Administration', 'Oceanography', 'Journalism and Communication',
    'Nuclear Science and Technology', 'Weapon Science and Technology', 'Naval Architecture and Ocean Engineering',
    'Environmental Science and Engineering', 'Transportation Engineering', 'Geology', 'Physical Oceanography',
    'Musicology', 'Stomatology', 'Aquaculture', 'Mechanical Engineering',
    'Aeronautical and Astronautical Science and Technology', 'Civil Engineering', 'Mechanics',
    'Petroleum and Natural Gas Engineering', 'Sociology', 'Food Science and Engineering', 'Agricultural Engineering',
    'Surveying and Mapping Science and Technology', 'Metallurgical Engineering',
    'Library, Information and Archival Management', 'Mining Engineering', 'Astronomy',
    'Geological Resources and Geological Engineering', 'Atmospheric Science', 'Optical Engineering', 'Animal Husbandry',
    'Geophysics', 'Crop Science', 'Management Science and Engineering', 'Psychology', 'Forestry',
    'Textile Science and Engineering', 'Veterinary Medicine', 'Instrument Science and Technology', 'Physical Education'
]

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


@Benchmark.register(
    name='super_gpqa',
    pretty_name='SuperGPQA',
    dataset_id='m-a-p/SuperGPQA',
    model_adapter=OutputType.GENERATION,
    output_types=[OutputType.MULTIPLE_CHOICE, OutputType.GENERATION],
    subset_list=SUBSET_LIST,
    metric_list=['AverageAccuracy'],
    few_shot_num=0,
    train_split=None,
    eval_split='train',  # only have train split
)
class SuperGPQAAdapter(DataAdapter):

    def __init__(self, **kwargs):
        few_shot_num = kwargs.get('few_shot_num', 0)
        if few_shot_num > 0 and few_shot_num != 5:
            logger.warning(
                f'Only support few_shot_num 0 or 5 for SuperGPQA, but got {few_shot_num}. Use 5-shot by default.')
            kwargs['few_shot_num'] = 5
        super().__init__(**kwargs)

        self.choices = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
        self.category_map = SUBSET_MAPPING
        self.few_shot_prompt = open(os.path.join(current_dir, 'five_shot_prompt.txt'), encoding='utf-8').read()
        self.zero_shot_prompt = open(os.path.join(current_dir, 'zero_shot_prompt.txt'), encoding='utf-8').read()

    def load(self, **kwargs):
        kwargs['subset_list'] = ['default']
        data_dict = super().load(**kwargs)
        return self.reformat_subset(data_dict, subset_key='field', format='{}')

    def gen_prompt(self, input_d: dict, subset_name: str, few_shot_list: list, **kwargs) -> dict:
        if not self.prompt_template:
            if few_shot_list:
                prompt = self.few_shot_prompt.format(query=input_d['question'])
            else:
                prompt = self.zero_shot_prompt.format(query=input_d['question'])
        else:
            prompt = self.prompt_template.format(query=input_d['question'])
        return self.gen_prompt_data(prompt)

    def get_gold_answer(self, input_d: dict) -> str:
        # Get the gold choice
        return input_d.get('answer_letter')

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
            from evalscope.benchmarks.super_gpqa.utils import extract_option_content, extract_option_labels
            sample = raw_input_d
            if self.few_shot_num == 0:
                predict = extract_option_labels(result, 'ABCDEFGHIJ')
                if predict is None:
                    predict = extract_option_content(result, sample['options'])
                    predict = chr(sample['options'].index(predict) + 65) if predict else None
            else:
                response = result.split('Question:')[0]
                predict = extract_option_labels(response, 'ABCDEFGHIJ')
                if predict is None:
                    predict = extract_option_content(response, sample['options'])
                    predict = chr(sample['options'].index(predict) + 65) if predict else None
                if predict is None:
                    predict = extract_option_labels(result, 'ABCDEFGHIJ')
                    if predict is None:
                        predict = extract_option_content(result, sample['options'])
                        predict = chr(sample['options'].index(predict) + 65) if predict else None
            return predict

    def match(self, gold: str, pred: str) -> float:
        return exact_match(gold=gold, pred=pred)
