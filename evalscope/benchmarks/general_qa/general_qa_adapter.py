# Copyright (c) Alibaba, Inc. and its affiliates.
import glob
import os.path
from collections import defaultdict
from typing import List

from evalscope.benchmarks import Benchmark, DataAdapter
from evalscope.metrics import bleu_ngram_one_sample, compute_rouge_score_one_sample_zh, mean
from evalscope.models import ChatGenerationModelAdapter
from evalscope.utils.io_utils import jsonl_to_list
from evalscope.utils.logger import get_logger

logger = get_logger()


@Benchmark.register(
    name='general_qa',
    dataset_id='general_qa',
    model_adapter=ChatGenerationModelAdapter,
    subset_list=['default'],
    metric_list=['AverageBLEU'],
    few_shot_num=0,
    train_split=None,
    eval_split='test',
)
class GeneralQAAdapter(DataAdapter):
    # TODO: set few_shot_num

    def __init__(self, **kwargs):

        super().__init__(**kwargs)

    def load(self, **kwargs) -> dict:

        data_file_list = glob.glob(os.path.join(self.dataset_id, '*.jsonl'))
        data_list = []

        try:
            for file_path in data_file_list:
                data_list.extend(jsonl_to_list(file_path))
        except Exception as e:
            raise ValueError(f'Failed to load data from {self.dataset_id}, got error: {e}')

        data_dict = {'default': {'test': data_list}}

        return data_dict

    def gen_prompt(self, input_d: dict, subset_name: str, few_shot_list: list, **kwargs) -> dict:
        """
        Args:
            input_d:
                format1: {'history': [['q1', 'a1'], ['q2', 'a2']], 'question': '', 'answer': ''}
                format2: {'history': [['q1', 'a1'], ['q2', 'a2']], 'query': '', 'response': ''}

        Returns:
            {'data': [prompt]}

        """
        # prompt = f"'<|im_start|>user\n{input_d['input']}<|im_end|>\n<|im_start|>assistant\n'"
        history = input_d.get('history', [])  # history: [['q1', 'a1'], ['q2', 'a2'], ...]
        if len(history) > 0:
            logger.warning('The history is not included in the prompt for GeneralQA. \
                           To be supported in the future.')

        prompt = input_d.get('question', '') or input_d.get('query', '')

        # if len(history) > 0:
        #     prompt = '\n'.join(history) + '\n' + prompt
        return {'data': [prompt], 'system_prompt': self.system_prompt}

    def get_gold_answer(self, input_d: dict) -> str:
        """
        Args:
            input_d: {'history': [], 'question': '', 'answer': ''}

        Returns:
            gold_answer: str

        """
        return input_d.get('answer', '') or input_d.get('response', '')

    def parse_pred_result(self, result: str, raw_input_d: dict = None, eval_type: str = 'checkpoint') -> str:
        """
        Args:
            result: str

        Returns:
            pred_result: str

        """
        return result

    def match(self, gold: str, pred: str) -> dict:
        """
        Args:
            gold: str
            pred: str

        Returns:
            bleu_score: dict

        """
        res = dict()
        rouge_dict = compute_rouge_score_one_sample_zh([pred], [gold])
        bleu_dict = bleu_ngram_one_sample(pred, gold)
        res.update(rouge_dict)
        res.update(bleu_dict)
        return res

    def compute_metric(self, review_res_list: List[dict]) -> List[dict]:
        """
        compute weighted mean of the bleu score of all samples

        Args:
            review_res_list: [score1, score2, ...]

        Returns:
            avg_res: List[dict]

        """
        items = defaultdict(list)
        for scores in review_res_list:
            for k, v in scores.items():
                items[k].append(v)
        # items = [(score, 1.0) for score in review_res_list]
        return [{'metric_name': k, 'score': mean(v), 'num': len(v)} for k, v in items.items()]
