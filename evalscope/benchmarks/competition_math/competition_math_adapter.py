# Copyright (c) Alibaba, Inc. and its affiliates.
# Copyright (c) EleutherAI, Inc. and its affiliates.
import glob
import json
import os
from collections import defaultdict

from evalscope.benchmarks import Benchmark, DataAdapter
from evalscope.metrics import extract_answer, math_equal, strip_answer_string
from evalscope.utils.logger import get_logger

# flake8: noqa

logger = get_logger()


@Benchmark.register(
    name='competition_math',
    pretty_name='MATH',
    dataset_id='modelscope/competition_math',
    subset_list=['Level 1', 'Level 2', 'Level 3', 'Level 4', 'Level 5'],
    metric_list=['AveragePass@1'],
    few_shot_num=4,
    train_split=None,
    eval_split='test',
    prompt_template='{query}\nPlease reason step by step, and put your final answer within \\boxed{{}}.',
)
class CompetitionMathAdapter(DataAdapter):
    """ To be tested for all models. """

    def __init__(self, **kwargs):

        few_shot_num = kwargs.get('few_shot_num', 4)
        if few_shot_num != 4 and few_shot_num != 0:
            logger.error(f'The MATH benchmark ONLY supports 4-shot by system or 0-shot settings, '
                         f'but got {few_shot_num}. Use 4-shot by default.')
            kwargs['few_shot_num'] = 4

        super().__init__(**kwargs)

    def load(self, **kwargs):
        # default load all levels
        kwargs['subset_list'] = ['default']
        data_dict = super().load(**kwargs)
        return self.reformat_subset(data_dict, subset_key='level')

    def load_from_disk(self, dataset_name_or_path, subset_list, work_dir, **kwargs) -> dict:
        data_dict = defaultdict(dict)
        for subset_name in subset_list:
            for split_name in [self.train_split, self.eval_split]:
                if os.path.exists(dataset_name_or_path):
                    split_dir = os.path.join(dataset_name_or_path, split_name)
                else:
                    split_dir = os.path.join(work_dir, dataset_name_or_path, split_name)
                split_files = glob.glob(os.path.join(split_dir, '**', '*.json'))
                split_data = []
                for file_path in split_files:
                    if os.path.exists(file_path):
                        with open(file_path, 'r', encoding='utf-8') as f:
                            split_data.append(json.load(f))
                data_dict[subset_name][split_name] = split_data

        return data_dict

    def gen_prompt(self, input_d: dict, few_shot_list: list, **kwargs) -> dict:
        """
        Generate the prompt for the model input.

        Args:
            input_d: raw input dict.
                {"problem": "How many vertical asymptotes does the graph of $y=\\frac{2}{x^2+x-6}$ have?", "level": "Level 3", "type": "Algebra", "solution": "The denominator of the rational function factors into $x^2+x-6=(x-2)(x+3)$. Since the numerator is always nonzero, there is a vertical asymptote whenever the denominator is $0$, which occurs for $x = 2$ and $x = -3$.  Therefore, the graph has $\\boxed{2}$ vertical asymptotes."}

            few_shot_list:  few shot list. Each item is a raw input dict.
            **kwargs:

        Returns:
            {'data': [prompt]}
        """
        use_fewshot = self.few_shot_num > 0
        query = self._generate_prompt(input_d, use_fewshot=use_fewshot)
        full_prompt = self.prompt_template.format(query=query)
        return self.gen_prompt_data(full_prompt)

    def get_gold_answer(self, input_d: dict) -> str:
        # Extract the gold answer from the input dict.
        return strip_answer_string(extract_answer(input_d['solution']))

    def parse_pred_result(self, result: str, raw_input_d: dict = None, eval_type: str = 'checkpoint') -> str:
        """
        Parse the model output to get the answer. Could be the best choice index.

        Args:
            result: Predicted answer from the model. Usually a string for chat.
            raw_input_d (dict): The raw input. Depending on the dataset.
            eval_type: 'checkpoint' or 'service' or `custom`

        Returns:
            The parsed answer. Depending on the dataset. Usually a string for chat.
        """
        # Note: Use same extraction method for both of checkpoint/service/custom
        result = strip_answer_string(extract_answer(result))
        return result

    def match(self, gold: str, pred: str) -> float:
        return math_equal(pred, gold)

    @classmethod
    def _generate_prompt(cls, input_d: dict, use_fewshot: bool = True) -> str:
        problem: str = input_d['problem']

        if use_fewshot:
            # Use 4-shot examples by system
            context = (
                'Problem:\nFind the domain of the expression $\\frac{{\sqrt{{x-2}}}}{{\sqrt{{5-x}}}}$.}}\nSolution:\nThe expressions inside each square root must be non-negative. Therefore, $x-2 \ge 0$, so $x\ge2$, and $5 - x \ge 0$, so $x \le 5$. Also, the denominator cannot be equal to zero, so $5-x>0$, which gives $x<5$. Therefore, the domain of the expression is $\\boxed{{[2,5)}}$.\nFinal Answer: The final answer is $[2,5)$. I hope it is correct.\n'
                'Problem:\nIf $\det \mathbf{{A}} = 2$ and $\det \mathbf{{B}} = 12,$ then find $\det (\mathbf{{A}} \mathbf{{B}}).$\nSolution:\nWe have that $\det (\mathbf{{A}} \mathbf{{B}}) = (\det \mathbf{{A}})(\det \mathbf{{B}}) = (2)(12) = \\boxed{{24}}.$\nFinal Answer: The final answer is $24$. I hope it is correct.\n'
                'Problem:\nTerrell usually lifts two 20-pound weights 12 times. If he uses two 15-pound weights instead, how many times must Terrell lift them in order to lift the same total weight?\nSolution:\nIf Terrell lifts two 20-pound weights 12 times, he lifts a total of $2\cdot 12\cdot20=480$ pounds of weight. If he lifts two 15-pound weights instead for $n$ times, he will lift a total of $2\cdot15\cdot n=30n$ pounds of weight. Equating this to 480 pounds, we can solve for $n$: \\begin{{align*}} 30n&=480\\\\ \Rightarrow\qquad n&=480/30=\\boxed{{16}} \end{{align*}}\nFinal Answer: The final answer is $16$. I hope it is correct.\n'
                'Problem:\nIf the system of equations: \\begin{{align*}} 6x-4y&=a,\\\\ 6y-9x &=b. \end{{align*}}has a solution $(x, y)$ where $x$ and $y$ are both nonzero, find $\\frac{{a}}{{b}},$ assuming $b$ is nonzero.\nSolution:\nIf we multiply the first equation by $-\\frac{{3}}{{2}}$, we obtain $$6y-9x=-\\frac{{3}}{{2}}a.$$Since we also know that $6y-9x=b$, we have $$-\\frac{{3}}{{2}}a=b\Rightarrow\\frac{{a}}{{b}}=\\boxed{{-\\frac{{2}}{{3}}}}.$$\nFinal Answer: The final answer is $-\\frac{{2}}{{3}}$. I hope it is correct.\n'
                f'Problem:\n{problem}\nSolution:\n')
        else:
            context = 'Problem:\n' + problem + '\nSolution:\n'
        return context
