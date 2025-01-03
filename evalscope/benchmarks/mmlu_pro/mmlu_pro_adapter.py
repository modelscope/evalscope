from collections import defaultdict
from typing import Any, Dict

from evalscope.benchmarks import Benchmark, DataAdapter
from evalscope.constants import AnswerKeys, EvalType
from evalscope.metrics import WeightedAverageAccuracy, exact_match
from evalscope.models import ChatGenerationModelAdapter
from evalscope.utils.utils import ResponseParser


@Benchmark.register(
    name='mmlu_pro',
    dataset_id='modelscope/mmlu-pro',
    model_adapter=ChatGenerationModelAdapter,
    subset_list=['default'],
    metric_list=[WeightedAverageAccuracy],
    few_shot_num=5,
    train_split='validation',
    eval_split='test',
    prompt_template=
    'You are an knowledge expert, you are supposed to answer the multi-choice question to derive your final answer as `The answer is ...`.',  # noqa: E501
)
class MMLUProAdapter(DataAdapter):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.choices = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
        self.categories = [
            'computer science', 'math', 'chemistry', 'engineering', 'law', 'biology', 'health', 'physics', 'business',
            'philosophy', 'economics', 'other', 'psychology', 'history'
        ]

    def gen_prompts(self, data_dict: dict, **kwargs) -> Dict[str, list]:
        """
        Generate model prompt from raw input, unify the prompt format for MMLU-Pro benchmark.
        Return a dict with category as key and list of prompts as value.
        """

        data_dict = data_dict[self.subset_list[0]]  # Only one subset for MMLU-Pro
        fewshot_prompts = self.get_fewshot_examples(data_dict)

        #  Use the category as key to group the prompts
        res_dict = defaultdict(list)
        # generate prompts for each test sample
        for entry in data_dict[self.eval_split]:
            prefix = fewshot_prompts[entry['category']]
            query = prefix + 'Q: ' + entry['question'] + '\n' + \
                self.__form_options(entry['options']) + '\n'

            prompt_d = {'data': [query], 'system_prompt': self.prompt_template, AnswerKeys.RAW_INPUT: entry}

            res_dict[entry['category']].append(prompt_d)
        return res_dict

    def get_fewshot_examples(self, data_dict: dict):
        # load 5-shot prompts for each category
        prompts = {c: '' for c in self.categories}
        for d in data_dict[self.train_split]:
            prompts[d['category']] += 'Q:' + ' ' + d['question'] + '\n' + \
                self.__form_options(d['options']) + '\n' + \
                d['cot_content'] + '\n\n'
        return prompts

    def __form_options(self, options: list):
        option_str = 'Options are:\n'
        for opt, choice in zip(options, self.choices):
            option_str += f'({choice}): {opt}' + '\n'
        return option_str

    def get_gold_answer(self, input_d: dict) -> str:
        """
        Parse the raw input labels (gold).

        Args:
            input_d: input raw data. Depending on the dataset.

        Returns:
            The parsed input. e.g. gold answer ... Depending on the dataset.
        """
        return input_d['answer']

    def parse_pred_result(self, result: str, raw_input_d: dict = None, eval_type: str = EvalType.CHECKPOINT) -> str:
        """
        Parse the predicted result and extract proper answer.

        Args:
            result: Predicted answer from the model. Usually a string for chat.
            raw_input_d: The raw input. Depending on the dataset.
            eval_type: 'checkpoint' or 'service' or `custom`, default: 'checkpoint'

        Returns:
            The parsed answer. Depending on the dataset. Usually a string for chat.
        """
        return ResponseParser.parse_first_option(result)

    def match(self, gold: str, pred: str) -> float:
        """
        Match the gold answer and the predicted answer.

        Args:
            gold (Any): The golden answer. Usually a string for chat/multiple-choice-questions.
                        e.g. 'A', extracted from get_gold_answer method.
            pred (Any): The predicted answer. Usually a string for chat/multiple-choice-questions.
                        e.g. 'B', extracted from parse_pred_result method.

        Returns:
            The match result. Usually a score (float) for chat/multiple-choice-questions.
        """
        return exact_match(gold=gold, pred=pred)
