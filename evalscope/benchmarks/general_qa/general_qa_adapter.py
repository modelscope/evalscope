# Copyright (c) Alibaba, Inc. and its affiliates.
import os.path
from collections import defaultdict
from typing import List, Optional, Union

from evalscope.benchmarks import Benchmark, DataAdapter
from evalscope.metrics import mean
from evalscope.utils.io_utils import jsonl_to_list
from evalscope.utils.logger import get_logger

logger = get_logger()


@Benchmark.register(
    name='general_qa',
    dataset_id='general_qa',
    subset_list=['default'],
    metric_list=['AverageBLEU', 'AverageRouge'],
    few_shot_num=0,
    train_split=None,
    eval_split='test',
    prompt_template='请回答问题\n{query}',
)
class GeneralQAAdapter(DataAdapter):
    # TODO: set few_shot_num

    def __init__(self, **kwargs):

        super().__init__(**kwargs)

    def load(self, dataset_name_or_path: str = None, subset_list: list = None, **kwargs) -> dict:
        dataset_name_or_path = dataset_name_or_path or self.dataset_id
        subset_list = subset_list or self.subset_list

        data_file_dict = defaultdict(str)
        data_item_dict = defaultdict(list)

        # get data file path and subset name
        if os.path.isdir(dataset_name_or_path):
            for subset_name in subset_list:
                data_file_dict[subset_name] = os.path.join(dataset_name_or_path, f'{subset_name}.jsonl')
        elif os.path.isfile(dataset_name_or_path):
            cur_subset_name = os.path.splitext(os.path.basename(dataset_name_or_path))[0]
            data_file_dict[cur_subset_name] = dataset_name_or_path
        else:
            raise ValueError(f'Invalid dataset path: {dataset_name_or_path}')

        # load data from local disk
        try:
            for subset_name, file_path in data_file_dict.items():
                data_item_dict[subset_name] = jsonl_to_list(file_path)
        except Exception as e:
            raise ValueError(f'Failed to load data from {self.dataset_id}, got error: {e}')

        data_dict = {subset_name: {'test': data_item_dict[subset_name]} for subset_name in data_file_dict.keys()}

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

        query = input_d.get('question', '') or input_d.get('query', '')
        system_prompt = input_d.get('system')
        prompt = self.prompt_template.format(query=query)
        return self.gen_prompt_data(prompt, system_prompt=system_prompt)

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
        if 'AverageRouge' in self.metric_list:
            from evalscope.metrics.rouge_metric import compute_rouge_score_one_sample_zh

            rouge_dict = compute_rouge_score_one_sample_zh([pred], [gold])
            res.update(rouge_dict)
        if 'AverageBLEU' in self.metric_list:
            from evalscope.metrics import bleu_ngram_one_sample

            bleu_dict = bleu_ngram_one_sample(pred, gold)
            res.update(bleu_dict)
        return res

    def compute_metric(self, review_res_list: Union[List[dict], List[List[dict]]], **kwargs) -> List[dict]:
        """
        compute weighted mean of the bleu score of all samples

        Args:
            review_res_list: [score1, score2, ...]

        Returns:
            avg_res: List[dict]

        """
        items = super().compute_dict_metric(review_res_list, **kwargs)
        return [{'metric_name': k, 'score': mean(v), 'num': len(v)} for k, v in items.items()]
