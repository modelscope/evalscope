# Copyright (c) Alibaba, Inc. and its affiliates.
# Copyright (c) EleutherAI Inc, and its affiliates.
import csv
import os

from evalscope.benchmarks import Benchmark
from evalscope.benchmarks.data_adapter import DataAdapter
from evalscope.constants import EvalType, OutputType
from evalscope.utils import get_logger

# flake8: noqa

logger = get_logger()


@Benchmark.register(
    name='trivia_qa',
    pretty_name='TriviaQA',
    dataset_id='modelscope/trivia_qa',
    subset_list=['default'],
    metric_list=['AverageAccuracy'],
    few_shot_num=5,
    train_split='dev',
    eval_split='test',
)
class TriviaQaAdapter(DataAdapter):

    def __init__(self, **kwargs):

        super().__init__(**kwargs)

    def load_from_disk(self, dataset_name_or_path, subset_list, work_dir, **kwargs) -> dict:
        data_dict = {}
        for subset_name in subset_list:
            data_dict[subset_name] = {}
            for split in [self.train_split, self.eval_split]:
                if os.path.exists(dataset_name_or_path):
                    file_path = os.path.join(dataset_name_or_path, f'trivia-{split}.qa.csv')
                else:
                    file_path = os.path.join(work_dir, dataset_name_or_path, f'trivia-{split}.qa.csv')
                if os.path.exists(file_path):
                    with open(file_path, 'r', encoding='utf-8') as f:
                        reader = csv.reader(f, delimiter='\t')
                        split_data = []
                        for row in reader:
                            assert len(row) == 2
                            question = row[0]
                            answers = eval(row[1])
                            split_data.append({
                                'input': [{
                                    'role': 'system',
                                    'content': 'Follow the given examples and answer the question.'
                                }, {
                                    'role': 'user',
                                    'content': question
                                }],
                                'ideal':
                                answers
                            })
                        data_dict[subset_name][split] = split_data

        return data_dict

    def gen_prompt(self, input_d: dict, subset_name: str, few_shot_list: list, **kwargs) -> dict:
        """
        Generate model prompt from raw input, unify the prompt format for TriviaQA benchmark.

        Args:
            input_d (dict): The raw input. A single data format of the TriviaQA:

            {
                "input": [
                    {"role": "system", "content": "Follow the given examples and answer the question."},
                    {"role": "user", "content": "Which Lloyd Webber musical premiered in the US on 10th December 1993?"}
                ],
                "ideal": [
                    "Sunset Blvd",
                    "West Sunset Boulevard",
                    "Sunset Boulevard",
                    "Sunset Bulevard",
                    "Sunset Blvd.",
                    "sunset boulevard",
                    "sunset bulevard",
                    "west sunset boulevard",
                    "sunset blvd"
                ]
            }

        Returns:
            {'data': [(context, continuation), ...]}
        """

        def get_sys_prompt(inp: dict) -> str:
            return inp['input'][0]['content']

        prompt = get_sys_prompt(input_d)
        few_shot_prompts = [self._generate_prompt(input_d=sample, include_answer=True) for sample in few_shot_list]
        context: str = '\n'.join(few_shot_prompts) + '\n'
        context += self._generate_prompt(input_d=input_d, include_answer=False)
        full_prompt = context

        return self.gen_prompt_data(full_prompt)

    def get_gold_answer(self, input_d: dict) -> list:
        # Get the gold choice
        ans: list = input_d.get('ideal', [])
        return ans

    def parse_pred_result(self, result: str, raw_input_d: dict = None, eval_type: str = EvalType.CHECKPOINT) -> str:
        """
        Parse the model output to get the answer.

        Args:
            result: Predicted answer from the model. A list of loglikelihood values for inputs pairs.
            raw_input_d: The raw input. A single data format of the TriviaQA:
            eval_type: The type of evaluation, e.g. 'checkpoint' or 'service' or 'custom'.

        Returns:
            The predicted answer.
        """
        return result

    def match(self, gold: list, pred: str) -> float:
        is_correct = any([cand in pred for cand in gold])
        return 1 if is_correct else 0

    @classmethod
    def _generate_prompt(cls, input_d: dict, include_answer=True) -> str:

        example: str = f"Question: {input_d['input'][1]['content']}\nAnswer:"
        if include_answer:
            example += f" {input_d['ideal'][0]}\n\n"

        return example
