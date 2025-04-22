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


@Benchmark.register(
    name='general_mcq',
    pretty_name='General MCQ',
    dataset_id='general_mcq',
    model_adapter=OutputType.GENERATION,
    output_types=[OutputType.MULTIPLE_CHOICE, OutputType.GENERATION],
    subset_list=['default'],
    metric_list=['AverageAccuracy'],
    few_shot_num=0,
    train_split='dev',
    eval_split='val',
    prompt_template='请回答问题，并选出其中的正确答案\n{query}',
    query_template='问题：{question}\n{choices}\n答案: {answer}\n\n')
class GeneralMCQAdapter(DataAdapter):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.choices = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']

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
            'answer': 'D'}

        Returns:
            {'data': ['prompt ...']}
        """

        few_shot_prompts = [self._format_example(input_d=sample, include_answer=True) for sample in few_shot_list]

        if len(few_shot_prompts) > 0:
            context: str = '\n'.join(few_shot_prompts) + '\n'
        else:
            context = ''
        context = context.strip() + self._format_example(input_d=input_d, include_answer=False)

        full_prompt = self.prompt_template.format(query=context)

        return self.gen_prompt_data(full_prompt)

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
        if self.model_adapter == OutputType.MULTIPLE_CHOICE:
            return result
        else:
            return ResponseParser.parse_first_option_with_choices(text=result, options=self.choices)

    def match(self, gold: str, pred: str) -> float:
        return exact_match(gold=gold, pred=pred)

    def _format_example(self, input_d: dict, include_answer=True):
        choices_str = '\n'.join([f'{choice}. {input_d[choice]}' for choice in self.choices if choice in input_d])

        if include_answer:
            return self.query_template.format(
                question=input_d['question'], choices=choices_str, answer=input_d['answer'])
        else:
            return self.query_template.format(question=input_d['question'], choices=choices_str, answer='').rstrip()
