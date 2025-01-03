# Copyright (c) Alibaba, Inc. and its affiliates.

import json
import os

from evalscope.benchmarks import Benchmark, DataAdapter
from evalscope.constants import EvalType
from evalscope.metrics import WeightedAverageAccuracy, exact_match
from evalscope.models import MultiChoiceModelAdapter
from evalscope.utils import ResponseParser
from evalscope.utils.logger import get_logger

# flake8: noqa

logger = get_logger()


@Benchmark.register(
    name='arc',
    dataset_id='modelscope/ai2_arc',
    model_adapter=MultiChoiceModelAdapter,
    subset_list=['ARC-Easy', 'ARC-Challenge'],
    metric_list=[WeightedAverageAccuracy],
    few_shot_num=0,
    train_split='train',
    eval_split='test',
    prompt_template='',
)
class ARCAdapter(DataAdapter):

    choices = ['A', 'B', 'C', 'D']

    def __init__(self, **kwargs):
        few_shot_num = kwargs.get('few_shot_num', None)
        if few_shot_num is None:
            # Use 0-shot by default
            logger.info(f'Set 0-shot examples by system for ARC.')
            few_shot_num = 0

        if few_shot_num != 0:
            logger.warning(f'few_shot_num is recommended to set 0 for ARC, got {few_shot_num}.')

        super().__init__(**kwargs)

    def load_from_disk(self, dataset_name_or_path, subset_list, work_dir, **kwargs) -> dict:
        """
        Load the dataset from local disk.

        dataset_name_or_path: str, the dataset id or path. e.g. 'arc'
        subset_list: list, the subset list to load. e.g. ['ARC-Easy', 'ARC-Challenge']
        work_dir: str, the local root data directory. e.g. '/path/to/data'
        kwargs: dict, other arguments.
        """
        data_dict = {}
        for subset_name in subset_list:
            if os.path.exists(dataset_name_or_path):
                subset_path = os.path.join(dataset_name_or_path, subset_name)
            else:
                subset_path = os.path.join(work_dir, dataset_name_or_path, subset_name)
            for split_name in ['Train', 'Test']:
                split_path = os.path.join(subset_path, f'{subset_name}-{split_name}.jsonl')
                if os.path.exists(split_path):
                    with open(split_path, 'r', errors='ignore') as in_f:
                        rows = []
                        for line in in_f:
                            item = json.loads(line.strip())
                            raw_choices = item['question']['choices']
                            rows.append({
                                'id': item['id'],
                                'question': item['question']['stem'],
                                'choices': {
                                    'text': [d['text'] for d in raw_choices],
                                    'label': [d['label'] for d in raw_choices]
                                },
                                'answerKey': item['answerKey'],
                            })

                        if subset_name in data_dict:
                            data_dict[subset_name].update({split_name.lower(): rows})
                        else:
                            data_dict[subset_name] = {split_name.lower(): rows}

        return data_dict

    def gen_prompt(self, input_d: dict, few_shot_list: list, **kwargs) -> dict:
        """
        Generate model prompt from raw data, unify the prompt format for ARC benchmark.

        Args:
            input_d (dict): The raw input. A single data format of the ARC:

            {
                'id': 'Mercury_7220990',
                'question': 'Which factor will most likely cause a person to develop a fever?',
                'choices':
                    {
                        'text':['a leg muscle relaxing after exercise',
                                'a bacterial population in the bloodstream',
                                'several viral particles on the skin',
                                'carbohydrates being digested in the stomach'],
                        'label': ['A', 'B', 'C', 'D']
                    },
                'answerKey': 'B'
            }

        Returns:
            {'data': ['xxx'], 'multi_choices': ['A', 'B', 'C', 'D']}
        """
        few_shot_prompts = [self._generate_prompt(input_d=sample, include_answer=True) for sample in few_shot_list]
        context: str = '\n'.join(few_shot_prompts)

        context = f'{self.prompt_template}\n{context}' if self.prompt_template else context

        # context = f'The following are multiple choice questions, please output correct answer in the form of A or B or C or D, do not output explanation:\n {context}'
        full_prompt: str = context + self._generate_prompt(input_d=input_d, include_answer=False)

        return {'data': [full_prompt], 'multi_choices': self.choices}

    def get_gold_answer(self, input_d: dict) -> str:
        # Get the gold choice
        return input_d.get('answerKey', '')

    def parse_pred_result(self, result: str, raw_input_d: dict = None, eval_type: str = EvalType.CHECKPOINT) -> str:
        """
        Parse the model output to get the answer. Could be the best choice index.

        Args:
            result: Predicted answer from the model. Usually a string for chat.
            raw_input_d (dict): The raw input. Depending on the dataset.
            eval_type: 'checkpoint' or 'service' or `custom`, default: 'checkpoint'

        Returns:
            The parsed answer. Depending on the dataset. Usually a string for chat.
        """
        if eval_type == EvalType.CHECKPOINT:
            return result
        elif eval_type == EvalType.SERVICE:
            return ResponseParser.parse_first_option_with_choices(
                text=result, options=self.choices)  # TODO: to be checked !
        elif eval_type == EvalType.CUSTOM:
            return ResponseParser.parse_first_option_with_choices(
                text=result, options=self.choices)  # TODO: to be checked !
        else:
            raise ValueError(f'Invalid eval_type: {eval_type}')

    def match(self, gold: str, pred: str) -> float:
        return exact_match(gold=gold, pred=pred)

    @classmethod
    def _generate_prompt(cls, input_d: dict, include_answer=True) -> str:

        example: str = input_d['question']

        choices_texts: list = input_d['choices']['text']
        choices_labels: list = input_d['choices']['label']
        choices_prompts: str = '\n'.join([label + '. ' + text for text, label in zip(choices_texts, choices_labels)])
        example += '\n' + choices_prompts

        example += '\nAnswer:'
        if include_answer:
            example += ' {}\n\n'.format(input_d['answerKey'])

        return example
