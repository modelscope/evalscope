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
    pretty_name='General-QA',
    description='A general question answering dataset for custom evaluation. '
    'For detailed instructions on how to use this benchmark, please refer to the [User Guide](https://evalscope.readthedocs.io/zh-cn/latest/advanced_guides/custom_dataset/llm.html#qa).',  # noqa: E501
    tags=['QA', 'Custom'],
    dataset_id='general_qa',
    subset_list=['default'],
    metric_list=['AverageBLEU', 'AverageRouge'],
    few_shot_num=0,
    train_split=None,
    eval_split='test',
    prompt_template='请回答问题\n{query}',
)
class GeneralQAAdapter(DataAdapter):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def load(self, dataset_name_or_path: str = None, subset_list: list = None, **kwargs) -> dict:
        """
        Load dataset from the given path or dataset name.

        Args:
            dataset_name_or_path (str): Path to dataset directory or file.
            subset_list (list): List of subset names to load.

        Returns:
            dict: Loaded dataset organized by subset.
        """
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
        Generate prompt for the model based on input data.

        Args:
            input_d (dict): Input data dictionary.
            subset_name (str): Name of the subset.
            few_shot_list (list): List of few-shot examples.

        Returns:
            dict: Dictionary containing the generated prompt.
        """
        messages = input_d.get('messages')
        query = input_d.get('question', '') or input_d.get('query', '')
        system_prompt = input_d.get('system')
        prompt = self.prompt_template.format(query=query)
        return self.gen_prompt_data(prompt, system_prompt=system_prompt, messages=messages)

    def get_gold_answer(self, input_d: dict) -> str:
        """
        Extract the gold (reference) answer from the input data.

        Args:
            input_d (dict): Input data dictionary.

        Returns:
            str: Gold answer string.
        """
        return input_d.get('answer') or input_d.get('response')

    def parse_pred_result(self, result: str, raw_input_d: dict = None, eval_type: str = 'checkpoint') -> str:
        """
        Parse the prediction result.

        Args:
            result (str): Model prediction result.
            raw_input_d (dict, optional): Original input data.
            eval_type (str): Evaluation type.

        Returns:
            str: Parsed prediction result.
        """
        return result

    def match(self, gold: str, pred: str) -> dict:
        """
        Compute metric scores between gold and predicted answers.

        Args:
            gold (str): Gold answer.
            pred (str): Predicted answer.

        Returns:
            dict: Dictionary of computed metric scores.
        """
        # reference free metrics
        if gold is None:
            return {'AverageAccuracy': -1}

        # calculate rouge and bleu scores
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
        Compute weighted mean of the metric scores for all samples.

        Args:
            review_res_list (list): List of metric score dictionaries.

        Returns:
            list: List of dictionaries with averaged metric results.
        """
        items = super().compute_dict_metric(review_res_list, **kwargs)
        return [{'metric_name': k, 'score': mean(v), 'num': len(v)} for k, v in items.items()]
