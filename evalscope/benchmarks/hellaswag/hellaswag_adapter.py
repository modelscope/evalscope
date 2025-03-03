# Copyright (c) Alibaba, Inc. and its affiliates.
import numpy as np
import os
import re

from evalscope.benchmarks import Benchmark, DataAdapter
from evalscope.constants import EvalType, OutputType
from evalscope.metrics import exact_match
from evalscope.utils.io_utils import jsonl_to_list
from evalscope.utils.logger import get_logger
from evalscope.utils.utils import ResponseParser

# flake8: noqa

logger = get_logger()


@Benchmark.register(
    name='hellaswag',
    pretty_name='HellaSwag',
    dataset_id='modelscope/hellaswag',
    model_adapter=OutputType.CONTINUOUS,
    output_types=[OutputType.CONTINUOUS, OutputType.GENERATION],
    subset_list=['default'],
    metric_list=['AverageAccuracy'],
    few_shot_num=0,
    train_split='train',
    eval_split='validation',
    prompt_template=
    'Respond with the index of sentence that makes the most sense, chose from 0, 1, 2, 3, derive your final answer as `The answer is ...`.',  # noqa: E501
)
class HellaSwagAdapter(DataAdapter):

    def __init__(self, **kwargs):

        few_shot_num = kwargs.get('few_shot_num', 0)
        if few_shot_num != 0:
            logger.warning(f'few_shot_num should be 0 for HellaSwag, but got {few_shot_num}. Use 0-shot by default.')
            kwargs['few_shot_num'] = 0

        super().__init__(**kwargs)
        self.choices = ['0', '1', '2', '3']

    def load_from_disk(self, dataset_name_or_path, subset_list, work_dir, **kwargs) -> dict:
        data_dict = {}
        for subset_name in subset_list:
            data_dict[subset_name] = {}
            for split in [self.train_split, self.eval_split]:
                if os.path.exists(dataset_name_or_path):
                    file_path = os.path.join(dataset_name_or_path, f'hellaswag_{split}.jsonl')
                else:
                    file_path = os.path.join(work_dir, dataset_name_or_path, f'hellaswag_{split}.jsonl')
                if os.path.exists(file_path):
                    data_dict[subset_name][split] = jsonl_to_list(file_path)

        return data_dict

    def gen_prompt(self, input_d: dict, few_shot_list: list, **kwargs) -> dict:
        """
        Generate model prompt from raw data, unify the prompt format for HellaSwag benchmark.

        Args:
            input_d (dict): The raw input. A single data format of the HellaSwag:

            {
                'ind': 4,
                'activity_label': 'Removing ice from car',
                'ctx_a': 'Then, the man writes over the snow covering the window of a car, and a woman wearing winter clothes smiles.',
                'ctx_b': 'then',
                'ctx': 'Then, the man writes over the snow covering the window of a car, and a woman wearing winter clothes smiles. then',
                'endings': [', the man adds wax to the windshield and cuts it.', ', a person board a ski lift, while two men supporting the head of the person wearing winter clothes snow as the we girls sled.', ', the man puts on a christmas coat, knitted with netting.', ', the man continues removing the snow on his car.'],
                'source_id': 'activitynet~v_-1IBHYS3L-Y',
                'split': 'train',
                'split_type': 'indomain',
                'label': '3'
            }

        Returns:
            Refer to function: evalscope.benchmarks.data_adapter.DataAdapter.gen_prompt for details.
        """

        endings: list = [self._preprocess(ending) for ending in input_d['endings']]

        few_shot_prompts = [
            self._generate_prompt(input_d=sample, endings=endings, include_answer=True) for sample in few_shot_list
        ]
        context: str = '\n'.join(few_shot_prompts) + '\n'
        context += self._generate_prompt(input_d=input_d, endings=endings, include_answer=False)

        ctx_continuation_pair_list = [(context.strip(), ' ' + cont.strip()) for cont in endings]

        return self.gen_prompt_data(ctx_continuation_pair_list)

    def get_gold_answer(self, input_d: dict) -> str:
        # Get the gold choice
        return input_d['label']

    def parse_pred_result(self, result: list, raw_input_d: dict = None, eval_type: str = EvalType.CHECKPOINT) -> str:
        """
        Parse the model output to get the answer. Could be the best choice index.

        Args:
            result: Predicted answer from the model. Usually a string for chat.
            raw_input_d: The raw input dict.
            eval_type: The evaluation type. e.g. checkpoint, service, custom.

        Returns:
            The parsed answer. Depending on the dataset. Usually a string for chat.
        """
        if self.model_adapter == OutputType.CONTINUOUS:
            # answer: in the form of [-2.3, -4.5, ...], len of self.choices
            result = np.array(result)
            endings: list = [self._preprocess(ending) for ending in raw_input_d['endings']]
            completion_len = np.array([float(len(i)) for i in endings])
            best_choice_idx = np.argmax(result / completion_len)

            return str(best_choice_idx)
        else:
            return ResponseParser.parse_first_option(result)

    def match(self, gold: str, pred: str) -> float:
        return exact_match(gold=str(gold), pred=str(pred))

    @classmethod
    def _preprocess(cls, text):
        text = text.strip()
        text = text.replace(' [title]', '. ')
        text = re.sub('\\[.*?\\]', '', text)
        text = text.replace('  ', ' ')
        return text

    @classmethod
    def _generate_prompt(cls, input_d: dict, endings: list, include_answer=True) -> str:
        """
        Generate prompt for HellaSwag dataset.

        Args:
            input_d: a single data of the hellaswag.
            endings:  preprocessed endings
            include_answer: bool

        Returns:

        """

        ctx = input_d['ctx_a'] + ' ' + input_d['ctx_b'].capitalize()
        example: str = cls._preprocess(input_d['activity_label'] + ': ' + ctx)

        if include_answer:
            example += '{}\n\n'.format(endings[int(input_d['label'])])

        return example
