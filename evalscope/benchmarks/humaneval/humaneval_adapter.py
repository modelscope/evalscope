# Copyright (c) Alibaba, Inc. and its affiliates.
import re

from evalscope.benchmarks import Benchmark, DataAdapter
from evalscope.utils.logger import get_logger

logger = get_logger()

# Example:
# {"task_id": "HumanEval/0", "prompt": "from typing import List\n\n\ndef has_close_elements(numbers: List[float], threshold: float) -> bool:\n    \"\"\" Check if in given list of numbers, are any two numbers closer to each other than\n    given threshold.\n    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)\n    False\n    >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)\n    True\n    \"\"\"\n", "entry_point": "has_close_elements", "canonical_solution": "    for idx, elem in enumerate(numbers):\n        for idx2, elem2 in enumerate(numbers):\n            if idx != idx2:\n                distance = abs(elem - elem2)\n                if distance < threshold:\n                    return True\n\n    return False\n", "test": "\n\nMETADATA = {\n    'author': 'jt',\n    'dataset': 'test'\n}\n\n\ndef check(candidate):\n    assert candidate([1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.3) == True\n    assert candidate([1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.05) == False\n    assert candidate([1.0, 2.0, 5.9, 4.0, 5.0], 0.95) == True\n    assert candidate([1.0, 2.0, 5.9, 4.0, 5.0], 0.8) == False\n    assert candidate([1.0, 2.0, 3.0, 4.0, 5.0, 2.0], 0.1) == True\n    assert candidate([1.1, 2.2, 3.1, 4.1, 5.1], 1.0) == True\n    assert candidate([1.1, 2.2, 3.1, 4.1, 5.1], 0.5) == False\n\n"}  # noqa


@Benchmark.register(
    name='humaneval',
    pretty_name='HumanEval',
    dataset_id='modelscope/humaneval',
    subset_list=['openai_humaneval'],
    metric_list=['Pass@1'],
    few_shot_num=0,
    train_split=None,
    eval_split='test',
    prompt_template='Complete the following python code:\n{query}',
    extra_params={
        'num_workers': 4,
        'timeout': 4
    },
)
class HumanevalAdapter(DataAdapter):
    """
    A placeholder for humaneval adapter, see HumanevalEvaluator for implementation.
    """

    def __init__(self, **kwargs):
        try:
            from human_eval.data import stream_jsonl, write_jsonl
            from human_eval.evaluation import check_correctness
        except ImportError:
            raise ImportError('Please install human_eval:'
                              'https://github.com/openai/human-eval/tree/master#installation , '
                              'Note that you need to enable the execution code in the human_eval/execution.py first.')
        super().__init__(**kwargs)

        extra_params = kwargs.get('extra_params', {})
        self.k = [1]
        self.num_workers = extra_params.get('num_workers', 4)
        self.timeout = extra_params.get('timeout', 4)

        self.read_problems_func = stream_jsonl
        self.write_jsonl_func = write_jsonl
        self.eval_func = check_correctness

    def load_from_disk(self, dataset_name_or_path, subset_list, work_dir, **kwargs) -> dict:
        data_dict = {}
        for subset_name in subset_list:
            data_dict[subset_name] = {}
            # [{'task_id': '', 'prompt': '', 'entry_point': '', 'canonical_solution': '', 'test': ''}, ...]
            data_dict[subset_name][self.eval_split] = [task for task in self.read_problems_func(dataset_name_or_path)]

        return data_dict

    def gen_prompt(self, input_d: dict, few_shot_list: list, **kwargs) -> dict:
        """
        Generate prompt for the model.

        Args:
            input_d (dict): The raw input. A single data format of the Humaneval:
            {'task_id': '', 'prompt': '', 'entry_point': '', 'canonical_solution': '', 'test': ''}
        """
        query = input_d['prompt']
        full_prompt = self.prompt_template.format(query=query)

        return self.gen_prompt_data(full_prompt)

    @classmethod
    def _postprocess(cls, text: str) -> str:
        if '```' in text:
            blocks = re.findall(r'```(.*?)```', text, re.DOTALL)
            if len(blocks) == 0:
                text = text.split('```')[1]  # fall back to default strategy
            else:
                text = blocks[0]  # fetch the first code block
                if not text.startswith('\n'):  # in case starting with ```python
                    text = text[max(text.find('\n') + 1, 0):]
        if text.strip().startswith('from') or text.strip().startswith('import'):
            def_idx = text.find('def')
            if def_idx != -1:
                text = text[max(text.find('\n', def_idx) + 1, 0):]
        text = text.split('\n\n')[0]
        if text.strip().startswith('def'):
            text = '\n'.join(text.split('\n')[1:])
        if not text.startswith('    '):
            if text.startswith(' '):
                text = '    ' + text.lstrip()
            else:
                text = '\n'.join(['    ' + line for line in text.split('\n')])
        return text

    def parse_pred_result(self, result: str, raw_input_d: dict = None, eval_type: str = 'checkpoint') -> str:
        return self._postprocess(result)

    def get_gold_answer(self, input_d: dict) -> str:
        return input_d

    def match(self, gold: str, pred: str) -> float:
        res = self.eval_func(gold, pred, self.timeout)
        return float(res['passed'])
