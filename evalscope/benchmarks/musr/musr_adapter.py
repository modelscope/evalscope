import ast
from typing import Any

from evalscope.benchmarks import Benchmark, DataAdapter
from evalscope.constants import EvalType, OutputType
from evalscope.metrics import exact_match
from evalscope.utils.utils import ResponseParser


@Benchmark.register(
    name='musr',
    pretty_name='MuSR',
    dataset_id='AI-ModelScope/MuSR',
    model_adapter=OutputType.GENERATION,
    output_types=[OutputType.MULTIPLE_CHOICE, OutputType.GENERATION],
    subset_list=['murder_mysteries', 'object_placements', 'team_allocation'],
    metric_list=['AverageAccuracy'],
    few_shot_num=0,
    train_split=None,
    eval_split='test',
    prompt_template=
    '{narrative}\n\n{question}\n\n{choices}\nThink step by step and then finish your answer with \"the answer is (X)\" where X is the correct letter choice.',  # noqa: E501
)
class MuSRAdapter(DataAdapter):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.choices = ['A', 'B', 'C', 'D', 'E', 'F']

    def load(self, **kwargs):
        # default load all levels
        kwargs['split_as_subset'] = True
        data_dict = super().load(**kwargs)
        return data_dict

    def gen_prompt(self, input_d: dict, subset_name: str, few_shot_list: list, **kwargs) -> Any:

        choices = self.format_choice(ast.literal_eval(input_d['choices']))

        full_prompt = self.prompt_template.format(
            narrative=input_d['narrative'], question=input_d['question'], choices=choices)

        return self.gen_prompt_data(full_prompt)

    def format_choice(self, options: list):
        option_str = ''
        for opt, choice in zip(options, self.choices):
            option_str += f'({choice}): {opt}\n'
        return option_str

    def get_gold_answer(self, input_d: dict) -> str:
        """
        Parse the raw input labels (gold).
        """
        return self.choices[input_d['answer_index']]

    def parse_pred_result(self, result: str, raw_input_d: dict = None, eval_type: str = EvalType.CHECKPOINT) -> str:
        """
        Parse the predicted result and extract proper answer.
        """
        if self.model_adapter == OutputType.MULTIPLE_CHOICE:
            return result
        else:
            return ResponseParser.parse_first_option(result, options=self.choices)

    def match(self, gold: str, pred: str) -> float:
        """
        Match the gold answer and the predicted answer.
        """
        return exact_match(gold=gold, pred=pred)
