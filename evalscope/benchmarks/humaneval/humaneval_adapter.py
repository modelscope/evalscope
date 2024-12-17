# Copyright (c) Alibaba, Inc. and its affiliates.
import json
import os
import re
from tqdm import tqdm
from typing import List

from evalscope.benchmarks.data_adapter import DataAdapter
from evalscope.metrics.metrics import weighted_mean
from evalscope.tools.combine_reports import gen_table
from evalscope.utils import normalize_score
from evalscope.utils.logger import get_logger

logger = get_logger()

DATASET_ID = 'modelscope/humaneval'
SUBSET_LIST = ['openai_humaneval']

# Example:
# {"task_id": "HumanEval/0", "prompt": "from typing import List\n\n\ndef has_close_elements(numbers: List[float], threshold: float) -> bool:\n    \"\"\" Check if in given list of numbers, are any two numbers closer to each other than\n    given threshold.\n    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)\n    False\n    >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)\n    True\n    \"\"\"\n", "entry_point": "has_close_elements", "canonical_solution": "    for idx, elem in enumerate(numbers):\n        for idx2, elem2 in enumerate(numbers):\n            if idx != idx2:\n                distance = abs(elem - elem2)\n                if distance < threshold:\n                    return True\n\n    return False\n", "test": "\n\nMETADATA = {\n    'author': 'jt',\n    'dataset': 'test'\n}\n\n\ndef check(candidate):\n    assert candidate([1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.3) == True\n    assert candidate([1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.05) == False\n    assert candidate([1.0, 2.0, 5.9, 4.0, 5.0], 0.95) == True\n    assert candidate([1.0, 2.0, 5.9, 4.0, 5.0], 0.8) == False\n    assert candidate([1.0, 2.0, 3.0, 4.0, 5.0, 2.0], 0.1) == True\n    assert candidate([1.1, 2.2, 3.1, 4.1, 5.1], 1.0) == True\n    assert candidate([1.1, 2.2, 3.1, 4.1, 5.1], 0.5) == False\n\n"}  # noqa


class HumanevalAdapter(DataAdapter):
    """
    A placeholder for humaneval adapter, see HumanevalEvaluator for implementation.
    """

    def __init__(self,
                 subset_list: list = None,
                 metric_list: list = None,
                 few_shot_num: int = None,
                 train_split: str = None,
                 eval_split: str = 'test',
                 prompt_template: str = 'Complete the following python code:\n',
                 **kwargs):
        try:
            from human_eval.data import stream_jsonl, write_jsonl
            from human_eval.evaluation import check_correctness
        except ImportError:
            raise ImportError('Please install human_eval:'
                              'https://github.com/openai/human-eval/tree/master#installation , '
                              'Note that you need to enable the execution code in the human_eval/execution.py first.')

        if subset_list is None:
            subset_list = SUBSET_LIST

        if metric_list is None:
            metric_list = [{'name': 'WeightedAverageAccuracy', 'object': weighted_mean}]

        self.k = [1]
        self.num_workers = 4
        self.timeout = 4.0
        self.outputs = kwargs.get('outputs', None)

        self.read_problems_func = stream_jsonl
        self.write_jsonl_func = write_jsonl
        self.eval_func = check_correctness

        super().__init__(
            subset_list=subset_list,
            metric_list=metric_list,
            few_shot_num=few_shot_num,
            train_split=train_split,
            eval_split=eval_split,
            prompt_template=prompt_template,
            **kwargs)

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
        full_prompt = input_d['prompt']
        full_prompt = f'{self.prompt_template}\n{full_prompt}' if self.prompt_template else full_prompt

        return {'data': [full_prompt]}

    def get_answers(self, infer_cfg: dict) -> List[dict]:
        ans_list: list = []
        system_prompt: str = ''
        for task_id, data_d in tqdm(self.problems.items(), total=len(self.problems), desc='Predicting(problems)'):
            prompt: str = system_prompt + data_d['prompt']
            inputs: dict = {'data': [prompt]}

            pred_res: dict = self.model_adapter.predict(inputs=inputs, infer_cfg=infer_cfg)

            pred_ans: str = pred_res['choices'][0]['message']['content']
            pred_ans = self._postprocess(pred_ans)

            ans_list.append({'task_id': task_id, 'completion': pred_ans})

        return ans_list

    def eval(self, infer_cfg: dict, **kwargs):

        # predict
        ans_list: list = self.get_answers(infer_cfg)
        ans_out_file: str = os.path.join(self.outputs_structure.predictions_dir, 'human_eval_predictions.jsonl')

        self.write_jsonl_func(filename=ans_out_file, data=ans_list)
        # logger.info(f'** Dump predictions to {ans_out_file} successfully.')
        logger.info('** Dump predictions successfully.')

        # evaluate  results: e.g. {'pass@1': 0.333, 'pass@10': 0.111}
        results = self.eval_func(
            sample_file=ans_out_file,
            k=self.k,
            n_workers=self.num_workers,
            timeout=self.timeout,
            problem_file=self.problem_file)

        # output: report
        report_map: dict = self.gen_report(results=results)
        report_dir: str = self.outputs_structure.reports_dir
        report_file: str = os.path.join(report_dir, 'human_eval_report.json')

        with open(report_file, 'w') as f:
            f.write(json.dumps(report_map, ensure_ascii=False, indent=4))
        # logger.info(f'** Dump report to {report_file} \n')
        logger.info('** Dump report \n')

        try:
            # Make table
            report_table: str = gen_table([report_dir])
            logger.info(f'** Report table: \n {report_table} \n')
        except Exception:
            logger.error('Failed to generate report table.')

    def gen_report(self, subset_score_map: dict, report_name: str = None) -> dict:
        total_num: int = sum([num for _, num in subset_score_map.values()])
        weighted_avg_acc: float = sum([score * num for score, num in subset_score_map.values()]) / total_num
        weighted_avg_acc = normalize_score(score=weighted_avg_acc)
        cate_avg_list = [{
            'name': subset_name,
            'score': normalize_score(score=score)
        } for subset_name, (score, _) in subset_score_map.items()]

        category_d = dict(name='DEFAULT', score=weighted_avg_acc, subset=cate_avg_list)

        res_map = dict(
            name=report_name or 'HumanEval',
            metric='pass@1',
            score=weighted_avg_acc,
            category=[category_d],
            total_num=total_num)

        return res_map

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

    def parse_pred_result(self, result: str, raw_input_d: dict = None, eval_type: str = 'checkpoint') -> str:
        return self._postprocess(result)

    def get_gold_answer(self, input_d: dict) -> str:
        return input_d

    def match(self, gold: str, pred: str) -> float:
        res = self.eval_func(gold, pred, self.timeout)
        return float(res['passed'])
