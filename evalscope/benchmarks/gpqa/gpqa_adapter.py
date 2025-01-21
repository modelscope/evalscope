import os
import random
import re

from evalscope.benchmarks import Benchmark, DataAdapter
from evalscope.constants import AnswerKeys, EvalType
from evalscope.metrics import Pass1, exact_match
from evalscope.models import ChatGenerationModelAdapter
from evalscope.utils.utils import ResponseParser


@Benchmark.register(
    name='gpqa',
    dataset_id='modelscope/gpqa',
    model_adapter=ChatGenerationModelAdapter,
    subset_list=['gpqa_extended', 'gpqa_main', 'gpqa_diamond'],
    metric_list=[Pass1],
    few_shot_num=5,
    train_split='train',
    eval_split='train',  # only have train split
    prompt_template='',
)
class GPQAAdapter(DataAdapter):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.choices = ['A', 'B', 'C', 'D']
        if self.few_shot_num and self.few_shot_num > 0:
            self.prompt_prefix = 'Here are some example questions from experts. Answer the final question yourself, following the format of the previous questions exactly.\n'  # noqa: E501
            self.prompt_prefix += open(os.path.join(os.path.dirname(__file__), 'chain_of_thought.txt'),
                                       'r').read() + '\nQuestion: '
        else:
            self.prompt_prefix = 'What is the correct answer to this question:'

    def gen_prompt(self, input_d: dict, subset_name: str, few_shot_list: list, **kwargs) -> dict:
        """
        Generate model prompt from input data.
        example:
        {
            "question":"天气预报说本周星期三会下雨，昨天果然下雨了，今天星期几？",
            "choice1":"星期一",
            "choice2":"星期二",
            "choice3":"星期三",
            "choice4":"星期四",
            "answer":"D",
        }
        """
        processed_input_d = self.__process_input(input_d)
        input_d['answer'] = processed_input_d['answer']  # add answer to input_d for answer extraction
        prompt = self.prompt_prefix + f"{input_d['Question']}\n{self.__form_options(processed_input_d['choices'])}Let's think step by step: "  # noqa: E501

        return {'data': [prompt], 'multi_choices': self.choices, 'system_prompt': self.prompt_template}

    def __process_input(self, input_d: dict) -> dict:

        def preprocess(text):
            if text is None:
                return ' '
            text = text.strip()
            text = text.replace(' [title]', '. ')
            text = re.sub('\\[.*?\\]', '', text)
            text = text.replace('  ', ' ')
            return text

        choices = [
            preprocess(input_d['Incorrect Answer 1']),
            preprocess(input_d['Incorrect Answer 2']),
            preprocess(input_d['Incorrect Answer 3']),
            preprocess(input_d['Correct Answer']),
        ]
        random.shuffle(choices)
        correct_answer_index = choices.index(preprocess(input_d['Correct Answer']))

        out_doc = {
            'choices': [choices[0], choices[1], choices[2], choices[3]],
            'answer': f"{chr(65 + correct_answer_index)}",
        }
        return out_doc

    def __form_options(self, options: list):
        option_str = 'Choices:\n'
        for opt, choice in zip(options, self.choices):
            option_str += f'({choice}) {opt}' + '\n'
        return option_str

    def get_gold_answer(self, input_d: dict) -> str:
        """
        Parse the raw input labels (gold).
        """
        return input_d['answer']

    def parse_pred_result(self, result: str, raw_input_d: dict = None, eval_type: str = EvalType.CHECKPOINT) -> str:
        """
        Parse the predicted result and extract proper answer.
        """
        return ResponseParser.parse_first_option_with_choices(result, self.choices)

    def match(self, gold: str, pred: str) -> float:
        """
        Match the gold answer and the predicted answer.
        """
        return exact_match(gold=gold, pred=pred)
