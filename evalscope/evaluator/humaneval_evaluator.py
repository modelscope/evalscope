import os
import re
from typing import List, Optional

import json
from tqdm import tqdm

from evalscope.constants import OutputsStructure
from evalscope.evaluator.evaluator import logger
from evalscope.models.model_adapter import BaseModelAdapter
from evalscope.tools.combine_reports import gen_table
from evalscope.utils import normalize_score


class HumanevalEvaluator(object):

    def __init__(
        self,
        problem_file: str,
        model_id: str,
        model_revision: str,
        model_adapter: BaseModelAdapter,
        outputs: Optional[OutputsStructure] = None,
        k: List[int] = [1, 10, 100],
        n_workers: int = 4,
        timeout: float = 3.0,
    ):
        try:
            from human_eval.evaluation import evaluate_functional_correctness
            from human_eval.data import read_problems, write_jsonl
        except ImportError:
            raise ImportError('Please install human_eval:'
                              'https://github.com/openai/human-eval/tree/master#installation , '
                              'Note that you need to enable the execution code in the human_eval/execution.py first.')

        self.problem_file = problem_file
        self.k = k
        self.num_workers = n_workers
        self.timeout = timeout
        self.model_adapter = model_adapter

        self.read_problems_func = read_problems
        self.write_jsonl_func = write_jsonl
        self.eval_func = evaluate_functional_correctness

        # {'task_id': {'task_id': '', 'prompt': '', 'entry_point': '', 'canonical_solution': '', 'test': ''}, ...}
        self.problems = self.read_problems_func(self.problem_file)

        # Deal with the output paths
        self.outputs_structure = OutputsStructure(outputs)

    def get_answers(self, infer_cfg: dict) -> List[dict]:
        ans_list: list = []
        system_prompt: str = 'Complete the following python code:\n'
        for task_id, data_d in tqdm(self.problems.items(), total=len(self.problems), desc='Predicting(problems)'):
            prompt: str = system_prompt + data_d['prompt']
            inputs: dict = {'data': [prompt]}
            # pred_res: dict = self.model_adapter.predict(inputs)

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

    def gen_report(self, results: dict) -> dict:
        """
        Generate report from evaluation results.

        Returns:
            {
            "name":"ARC-Challenge",
            "metric":"WeightedAverageAccuracy",
            "score":0.3389,
            "category":[
                {
                    "name":"DEFAULT",
                    "score":0.3389,
                    "subset":[
                        {
                            "name":"ARC-Challenge",
                            "score":0.3389
                        },
                    ]
                }
            ],
            "total_num":100
        }
        """
        results = {k: normalize_score(score=v) for k, v in results.items()}

        category_d = dict(name='DEFAULT', score=results, subset=[])

        res_map = dict(
            name='HumanEval', metric='pass@k', score=results, category=[category_d], total_num=len(self.problems))

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
