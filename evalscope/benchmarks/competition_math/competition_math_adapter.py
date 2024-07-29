# Copyright (c) Alibaba, Inc. and its affiliates.
# Copyright (c) EleutherAI, Inc. and its affiliates.
import glob
import json
import os

from evalscope.benchmarks import DataAdapter
from evalscope.metrics.metrics import weighted_mean
from evalscope.utils import normalize_score
from evalscope.utils.logger import get_logger
# flake8: noqa

logger = get_logger()

DATASET_ID = 'modelscope/competition_math'
SUBSET_LIST = ['default']


class CompetitionMathAdapter(DataAdapter):
    """ TODO: To be tested for all models. """

    def __init__(self,
                 subset_list: list = None,
                 metric_list: list = None,
                 few_shot_num: int = None,
                 train_split: str = 'train',
                 eval_split: str = 'test',
                 **kwargs):

        if subset_list is None:
            subset_list = SUBSET_LIST

        if metric_list is None:
            metric_list = [{'name': 'WeightedAverageAccuracy', 'object': weighted_mean}]

        if few_shot_num is None:
            # Use 4-shot by default
            logger.info(f'Set 4-shot examples by system for MATH.')
            few_shot_num = 4

        if few_shot_num != 4 and few_shot_num != 0:
            logger.error(f'The MATH benchmark ONLY supports 4-shot by system or 0-shot settings, '
                         f'but got {self.few_shot_num}. Use 4-shot by default.')
            few_shot_num = 4

        super().__init__(subset_list=subset_list,
                         metric_list=metric_list,
                         few_shot_num=few_shot_num,
                         train_split=train_split,
                         eval_split=eval_split,
                         **kwargs)

    def load_from_disk(self, dataset_name_or_path, subset_list, work_dir, **kwargs) -> dict:
        data_dict: dict = {}
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
                        with open(file_path, 'r') as f:
                            split_data.append(json.load(f))
                if subset_name in data_dict:
                    data_dict[subset_name].update({split_name: split_data})
                else:
                    data_dict[subset_name] = {split_name: split_data}

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
        full_prompt = self._generate_prompt(input_d, use_fewshot=use_fewshot)

        return {'data': [full_prompt]}

    def get_gold_answer(self, input_d: dict) -> str:
        # Extract the gold answer from the input dict.
        return self._preprocess_input(input_d['solution'])

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
        # TODO: check answer extraction
        # Note: Use same extraction method for both of checkpoint/service/custom
        return self._math_postprocess(result)

    def match(self, gold: str, pred: str) -> float:
        res = 0
        if self._is_equiv(pred, gold):
            res = 1

        return res

    def compute_metric(self, review_res_list: list) -> float:
        """
        Compute evaluation result by specific metric.

        Args:
            review_res_list: review score list, e.g. [0, 1, 1, 0, ...]

        Returns:
            The metric score.
        """
        items = [(score, 1.0) for score in review_res_list]
        return weighted_mean(items)

    def gen_report(self, subset_score_map: dict, report_name: str = None) -> dict:
        """
        Generate the report for the model output.

        Args:
            subset_score_map: The subset-score mapping. e.g. {subset_name: (score, num), ...}
            report_name: The user-defined report name.

        Returns: A dict of metric calculation results. The format is like:
        {
            "name":"CompetitionMath",
            "metric":"WeightedAverageAccuracy",
            "score":0.5632,
            "category":[
                {
                    "name":"DEFAULT",
                    "score":0.5632,
                    "subset":[
                        {
                            "name":"main",
                            "score":0.5632
                        },
                    ]
                }
            ],
            "total_num":100
        }
        """
        total_num: int = sum([num for _, num in subset_score_map.values()])
        weighted_avg_acc: float = sum([score * num for score, num in subset_score_map.values()]) / total_num
        weighted_avg_acc = normalize_score(score=weighted_avg_acc)
        cate_avg_list = [{'name': subset_name, 'score': normalize_score(score=score)} for subset_name, (score, _) in subset_score_map.items()]

        category_d = dict(name='DEFAULT',
                          score=weighted_avg_acc,
                          subset=cate_avg_list)

        res_map = dict(name=report_name or 'competition_math',
                       metric=self.metric_list[0]['name'],
                       score=weighted_avg_acc,
                       category=[category_d],
                       total_num=total_num)

        return res_map

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
                f'Problem:\n{problem}\nSolution:\n'
            )
        else:
            context = 'Problem:\n' + problem + '\nSolution:\n'
        return context

    @classmethod
    def _preprocess_input(cls, input: str) -> str:
        """
        Preprocess the input data, remove the boxed solution.

        Args:
            input_d: The raw input. A single data format of the Competition Math.

        Returns:
            The preprocessed input.
        """
        return cls._remove_boxed(cls._last_boxed_only_string(input))

    @classmethod
    def _remove_boxed(cls, s):
        if s is None:
            return s

        if '\\boxed ' in s:
            left = '\\boxed '
            assert s[: len(left)] == left
            return s[len(left):]

        left = '\\boxed{'

        assert s[: len(left)] == left
        assert s[-1] == '}'

        return s[len(left): -1]

    @classmethod
    def _last_boxed_only_string(cls, string):

        idx = string.rfind('\\boxed')
        if '\\boxed ' in string:
            return '\\boxed ' + string.split('\\boxed ')[-1].split('$')[0]
        if idx < 0:
            idx = string.rfind('\\fbox')
            if idx < 0:
                return None

        i = idx
        right_brace_idx = None
        num_left_braces_open = 0
        while i < len(string):
            if string[i] == '{':
                num_left_braces_open += 1
            if string[i] == '}':
                num_left_braces_open -= 1
                if num_left_braces_open == 0:
                    right_brace_idx = i
                    break
            i += 1

        if right_brace_idx is None:
            retval = None
        else:
            retval = string[idx: right_brace_idx + 1]

        return retval

    @classmethod
    def _is_equiv(cls, str1, str2, verbose=False):
        if str1 is None and str2 is None:
            logger.warning('WARNING: Both None')
            return True
        if str1 is None or str2 is None:
            return False

        try:
            ss1 = cls.strip_string(str1)
            ss2 = cls.strip_string(str2)
            if verbose:
                logger.info(f'ss1: {ss1}, ss2: {ss2}')
            return ss1 == ss2
        except Exception:
            return str1 == str2

    @classmethod
    def strip_string(cls, string):
        # linebreaks
        string = string.replace('\n', '')

        # remove inverse spaces
        string = string.replace('\\!', '')

        # replace \\ with \
        string = string.replace('\\\\', '\\')

        # replace tfrac and dfrac with frac
        string = string.replace('tfrac', 'frac')
        string = string.replace('dfrac', 'frac')

        # remove \left and \right
        string = string.replace('\\left', '')
        string = string.replace('\\right', '')

        # Remove circ (degrees)
        string = string.replace('^{\\circ}', '')
        string = string.replace('^\\circ', '')

        # remove dollar signs
        string = string.replace('\\$', '')

        # remove units (on the right)
        string = cls.remove_right_units(string)

        # remove percentage
        string = string.replace('\\%', '')
        string = string.replace('\%', '')  # noqa: W605

        # " 0." equivalent to " ." and "{0." equivalent to "{." Alternatively, add "0" if "." is the start of the string
        string = string.replace(' .', ' 0.')
        string = string.replace('{.', '{0.')
        # if empty, return empty string
        if len(string) == 0:
            return string
        if string[0] == '.':
            string = '0' + string

        # to consider: get rid of e.g. "k = " or "q = " at beginning
        if len(string.split('=')) == 2:
            if len(string.split('=')[0]) <= 2:
                string = string.split('=')[1]

        # fix sqrt3 --> sqrt{3}
        string = cls.fix_sqrt(string)

        # remove spaces
        string = string.replace(' ', '')

        # \frac1b or \frac12 --> \frac{1}{b} and \frac{1}{2}, etc. Even works with \frac1{72} (but not \frac{72}1). Also does a/b --> \\frac{a}{b}
        string = cls.fix_fracs(string)

        # manually change 0.5 --> \frac{1}{2}
        if string == '0.5':
            string = '\\frac{1}{2}'

        # NOTE: X/Y changed to \frac{X}{Y} in dataset, but in simple cases fix in case the model output is X/Y
        string = cls.fix_a_slash_b(string)

        return string

    @classmethod
    def remove_right_units(cls, string):
        # "\\text{ " only ever occurs (at least in the val set) when describing units
        if '\\text{ ' in string:
            splits = string.split('\\text{ ')
            assert len(splits) == 2
            return splits[0]
        else:
            return string

    @classmethod
    def fix_fracs(cls, string):
        substrs = string.split('\\frac')
        new_str = substrs[0]
        if len(substrs) > 1:
            substrs = substrs[1:]
            for substr in substrs:
                new_str += '\\frac'
                if substr[0] == '{':
                    new_str += substr
                else:
                    try:
                        assert len(substr) >= 2
                    except AssertionError:
                        return string
                    a = substr[0]
                    b = substr[1]
                    if b != '{':
                        if len(substr) > 2:
                            post_substr = substr[2:]
                            new_str += '{' + a + '}{' + b + '}' + post_substr
                        else:
                            new_str += '{' + a + '}{' + b + '}'
                    else:
                        if len(substr) > 2:
                            post_substr = substr[2:]
                            new_str += '{' + a + '}' + b + post_substr
                        else:
                            new_str += '{' + a + '}' + b
        string = new_str
        return string

    @classmethod
    def fix_sqrt(cls, string):
        if '\\sqrt' not in string:
            return string
        splits = string.split('\\sqrt')
        new_string = splits[0]
        for split in splits[1:]:
            if split[0] != '{':
                a = split[0]
                new_substr = '\\sqrt{' + a + '}' + split[1:]
            else:
                new_substr = '\\sqrt' + split
            new_string += new_substr
        return new_string

    @classmethod
    def fix_a_slash_b(cls, string):
        if len(string.split('/')) != 2:
            return string
        a = string.split('/')[0]
        b = string.split('/')[1]
        try:
            a = int(a)
            b = int(b)
            assert string == '{}/{}'.format(a, b)
            new_string = '\\frac{' + str(a) + '}{' + str(b) + '}'
            return new_string
        except AssertionError:
            return string

    @classmethod
    def _math_postprocess(cls, text: str) -> str:
        SUBSTITUTIONS = [('an ', ''), ('a ', ''), ('.$', '$'), ('\\$', ''),
                         (r'\ ', ''), (' ', ''), ('mbox', 'text'),
                         (',\\text{and}', ','), ('\\text{and}', ','),
                         ('\\text{m}', '\\text{}'), ('\\le', '<')]
        REMOVED_EXPRESSIONS = [
            'square', 'ways', 'integers', 'dollars', 'mph', 'inches', 'ft',
            'hours', 'km', 'units', '\\ldots', 'sue', 'points', 'feet', 'minutes',
            'digits', 'cents', 'degrees', 'cm', 'gm', 'pounds', 'meters', 'meals',
            'edges', 'students', 'childrentickets', 'multiples', '\\text{s}',
            '\\text{.}', '\\text{\ns}', '\\text{}^2', '\\text{}^3', '\\text{\n}',
            '\\text{}', r'\mathrm{th}', r'^\circ', r'^{\circ}', r'\;', r',\!',
            '{,}', '"', '\\dots', '\n', '\r', '\f'
        ]
        import re

        def normalize_final_answer(final_answer: str) -> str:
            """Normalize a final answer to a quantitative reasoning question."""
            # final_answer = final_answer.split('=')[-1]
            for before, after in SUBSTITUTIONS:
                final_answer = final_answer.replace(before, after)
            for expr in REMOVED_EXPRESSIONS:
                final_answer = final_answer.replace(expr, '')

            # Extract answer that is in LaTeX math, is bold,
            # is surrounded by a box, etc.
            final_answer = re.sub(r'(\\text\{)(.*?)(\})', '\\2', final_answer)
            final_answer = re.sub(r'(\\textbf\{)(.*?)(\})', '\\2', final_answer)
            final_answer = re.sub(r'(\\overline\{)(.*?)(\})', '\\2', final_answer)
            final_answer = re.sub(r'(\\boxed\{)(.*)(\})', '\\2', final_answer)
            assert '\n' not in final_answer
            assert '\r' not in final_answer
            assert '\f' not in final_answer
            if len(re.findall(r'finalansweris(.*)', final_answer)) > 0:
                final_answer = re.findall(r'finalansweris(.*)', final_answer)[-1]

            if len(re.findall(r'oxed\{(.*?)\}', final_answer)) > 0:
                final_answer = re.findall(r'oxed\{(.*?)\}', final_answer)[-1]

            if len(re.findall(r'\$(.*?)\$', final_answer)) > 0:
                final_answer = re.findall(r'\$(.*?)\$', final_answer)[-1]
            final_answer = final_answer.strip()
            if 'rac' in final_answer and '\\frac' not in final_answer:
                final_answer = final_answer.replace('rac', '\\frac')

            final_answer = re.sub(r'(frac)([^{])(.)', 'frac{\\2}{\\3}',
                                  final_answer)
            final_answer = re.sub(r'(sqrt)([^{])', 'sqrt{\\2}', final_answer)
            final_answer = final_answer.replace('$', '')

            # Normalize 100,000 -> 100000
            if final_answer.replace(',', '').isdigit():
                final_answer = final_answer.replace(',', '')

            return final_answer

        for maybe_ans in text.split('.'):
            if 'final answer' in maybe_ans.lower():
                return normalize_final_answer(maybe_ans)
        return normalize_final_answer(text.split('.')[0])
