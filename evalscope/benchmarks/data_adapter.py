# Copyright (c) Alibaba, Inc. and its affiliates.
import os.path
import random
from abc import ABC, abstractmethod
from typing import Any, Optional

from evalscope.constants import DEFAULT_DATASET_CACHE_DIR, AnswerKeys, EvalType, HubType
from evalscope.utils import normalize_score
from evalscope.utils.logger import get_logger

logger = get_logger()


class DataAdapter(ABC):

    def __init__(self,
                 subset_list: list,
                 metric_list: list,
                 few_shot_num: Optional[int] = 0,
                 train_split: Optional[str] = None,
                 eval_split: Optional[str] = None,
                 prompt_template: str = '',
                 **kwargs):
        """
        Data Adapter for the benchmark. You need to implement the following methods:
            - gen_prompt
            - get_gold_answer
            - parse_pred_result
            - match
        Args:
            subset_list: list of subset names for the dataset.
            metric_list: list, the metric list to evaluate the model on specific benchmark.
            few_shot_num: int, number of few-shot examples. Default: 0
            train_split: str, usually for few-shot examples. e.g. 'train'
            eval_split: str, the target eval split name. e.g. 'test'
            prompt_template: str, the prompt template for the benchmark,
                e.g. for ARC, it is `The following are multiple choice questions, please output correct answer in
                    the form of A or B or C or D, do not output explanation:`
        """
        self.subset_list = subset_list
        self.metric_list = metric_list
        self.few_shot_num = few_shot_num
        self.train_split = train_split
        self.eval_split = eval_split
        self.prompt_template = prompt_template
        self.config_kwargs = kwargs

    def load(self,
             dataset_name_or_path: str,
             subset_list: list = None,
             work_dir: Optional[str] = DEFAULT_DATASET_CACHE_DIR,
             datasets_hub: str = HubType.MODELSCOPE,
             **kwargs) -> dict:
        """
        Load the dataset. Remote and local datasets are supported.
        You can rewrite this method to support your own local dataset, just follow the format of the output.

        Returns: {'subset_name': {'train': train_dataset, 'test': test_dataset}}
            train_dataset, test_dataset: Iterable dataset, object each item of which is a dict.

        """
        dataset_name_or_path = os.path.expanduser(dataset_name_or_path)
        subset_list = subset_list or self.subset_list

        # Try to load dataset from local disk
        if os.path.exists(dataset_name_or_path):
            logger.info(f'Loading dataset from work_dir: {work_dir}: > dataset_name: {dataset_name_or_path} > \
                    subsets: {subset_list}')
            data_dict = self.load_from_disk(dataset_name_or_path, subset_list, work_dir, **kwargs)
            if len(data_dict) == 0 or len(next(iter(data_dict.values()))) == 0:
                raise ValueError(f'Local dataset is empty: {dataset_name_or_path}')
        else:
            from modelscope.msdatasets import MsDataset

            # Load dataset from remote
            logger.info(
                f'Loading dataset from {datasets_hub}: > dataset_name: {dataset_name_or_path} > subsets: {subset_list}')
            data_dict = {}
            split_list = [split for split in [self.train_split, self.eval_split] if split is not None]
            if len(split_list) == 0:
                logger.error(f'Got empty split list: {split_list}')

            for sub_name in subset_list:
                data_dict[sub_name] = {}
                # e.g. train: few-shot, test: target dataset to evaluate
                for split in split_list:
                    dataset = MsDataset.load(
                        dataset_name=dataset_name_or_path,
                        subset_name=sub_name,
                        split=split,
                        cache_dir=work_dir,
                        hub=datasets_hub,
                        **kwargs)

                    data_dict[sub_name].update({split: dataset})

        return data_dict

    def load_from_disk(self, *args, **kwargs) -> dict:
        """
        Load the dataset from local disk.
        If you want to support local dataset, please rewrite this method in xxx_data_adapter.
        """
        return {}

    def gen_prompts(self, data_dict: dict) -> dict:
        """
        Generate dataset prompts from raw input, unify the prompt format for different datasets.

        Args:
            data_dict:  Refer to the output of load method: evalscope.benchmarks.benchmark.Benchmark.load

        Returns:
            {'subset_name': [prompt_d_1, prompt_d_2, ...]}
            prompt_d_i (dict): refer to the output of gen_prompt method.

        e.g. train -- few-shot data, test -- target dataset to evaluate.
        """
        res_dict: dict = {}

        if self.few_shot_num and self.few_shot_num < 0:
            raise ValueError(f'Invalid shot_num: {self.few_shot_num} for few-shot evaluation.')

        logger.info(f'Use default settings: '
                    f'> few_shot_num: {self.few_shot_num}, '
                    f'> few_shot_split: {self.train_split}, '
                    f'> target_eval_split: {self.eval_split}')

        for sub_name, sub_data_dict in data_dict.items():
            few_shot_data = []
            if self.few_shot_num and self.few_shot_num > 0:
                few_shot_random: bool = self.config_kwargs.get('few_shot_random', True)
                few_shot_data = self.get_fewshot_examples([item for item in sub_data_dict[self.train_split]],
                                                          self.few_shot_num,
                                                          few_shot_random=few_shot_random)

            res_dict[sub_name] = []
            for sample_d in sub_data_dict[self.eval_split]:
                prompt_d = self.gen_prompt(input_d=sample_d, subset_name=sub_name, few_shot_list=few_shot_data)
                prompt_d[AnswerKeys.RAW_INPUT] = sample_d
                res_dict[sub_name].append(prompt_d)

        return res_dict

    def gen_report(self, subset_score_map: dict, report_name: str = None) -> dict:
        """
        Generate report for the evaluation results for all subsets.

        Args:
            subset_score_map: The subset-score map.
                e.g. {subset_name: (score, num)}

            report_name: str, the user-defined report name. Default: None

        Returns: The evaluation report.  Note: should normalize the score by normalize_score method in utils.

        Here is a format example for ARC-Challenge:
        {
            "name":"ARC-Challenge",
            "metric":"WeightedAverageAccuracy",
            "score": 0.3389,
            "category":[
                {
                    "name":"DEFAULT",
                    "score": 0.3389,
                    "subset":[
                        {
                            "name":"ARC-Challenge",
                            "score": 0.3389,
                            "num": 100
                        },
                    ]
                }
            ],
            "total_num":100
        }
        """  # noqa: E501
        total_num: int = sum([num for _, num in subset_score_map.values()])
        weighted_avg_acc: float = sum([score * num for score, num in subset_score_map.values()]) / total_num
        weighted_avg_acc = normalize_score(score=weighted_avg_acc)
        cate_avg_list = [{
            'name': subset_name,
            'score': normalize_score(score=score),
            'num': num
        } for subset_name, (score, num) in subset_score_map.items()]

        category_d = dict(name='DEFAULT', score=weighted_avg_acc, subset=cate_avg_list)

        res_map = dict(
            name=report_name or 'DEFAULT',
            metric=self.metric_list[0]['name'],
            score=weighted_avg_acc,
            category=[category_d],
            total_num=total_num)

        return res_map

    def get_fewshot_examples(self, data_list: list, k: int, few_shot_random: bool = True):

        if k > len(data_list):
            k = len(data_list)
        if few_shot_random:
            return random.sample(data_list, k)
        else:
            return data_list[:k]

    def compute_metric(self, review_res_list: list) -> Any:
        """
        Compute evaluation result by specific metrics.

        Args:
            review_res_list: list, the review result list, each item of which is match result for gold and pred.

        Attributes:
            DataAdapter.metric_func_map: metric_name -> metric_func mapping,
                e.g. {'WeightedAverageAccuracy': weighted_average_acc}

        Returns:
            Metric results.
        """
        if len(self.metric_list) == 0:
            raise ValueError('No metric list found for the benchmark.')
        elif len(self.metric_list) == 1:
            # review_res_list: review score list, e.g. [0, 1, 1, 0, ...]
            items = [(score, 1.0) for score in review_res_list]
            return self.metric_list[0]['object'](items)
        else:
            raise ValueError('Please implement the compute_metric method for multiple metrics.')

    def gen_prompt(self, input_d: dict, subset_name: str, few_shot_list: list, **kwargs) -> Any:
        """
        Generate model prompt from raw input, unify the prompt format for different datasets.
        The input format is compatible with OpenAI Chat Completions APIs.

        Args:
            input_d (Any): The raw input. Depending on the dataset.
            subset_name (str): The subset name.
            few_shot_list (list): The few-shot examples.

        Returns:
            For class ChatGenerationModelAdapter, the output format is:
                {'data': [full_prompt], 'system_prompt': (str, optional)},  -- full_prompt: str, the constructed prompt for each sample from dataset.
            For class MultiChoiceModelAdapter, the output format is:
                {'data': [full_prompt], 'multi_choices': self.choices}  -- full_prompt: str, the constructed prompt for each sample from dataset.
            For class ContinuationEvalModelAdapter, the output format is:
                {'data': ctx_continuation_pair_list, 'multi_choices': self.choices} -- ctx_continuation_pair_list: list, the context-continuation pair list.
        """  # noqa: E501
        raise NotImplementedError

    @abstractmethod
    def get_gold_answer(self, input_d: Any) -> Any:
        """
        Parse the raw input labels (gold).

        Args:
            input_d: input raw data. Depending on the dataset.

        Returns:
            The parsed input. e.g. gold answer ... Depending on the dataset.
        """
        raise NotImplementedError

    @abstractmethod
    def parse_pred_result(self, result: Any, raw_input_d: dict = None, eval_type: str = EvalType.CHECKPOINT) -> Any:
        """
        Parse the predicted result and extract proper answer.

        Args:
            result: Predicted answer from the model. Usually a string for chat.
            raw_input_d: The raw input. Depending on the dataset.
            eval_type: 'checkpoint' or 'service' or `custom`, default: 'checkpoint'

        Returns:
            The parsed answer. Depending on the dataset. Usually a string for chat.
        """
        raise NotImplementedError

    @abstractmethod
    def match(self, gold: Any, pred: Any) -> float:
        """
        Match the gold answer and the predicted answer.

        Args:
            gold (Any): The golden answer. Usually a string for chat/multiple-choice-questions.
                        e.g. 'A', extracted from get_gold_answer method.
            pred (Any): The predicted answer. Usually a string for chat/multiple-choice-questions.
                        e.g. 'B', extracted from parse_pred_result method.

        Returns:
            The match result. Usually a score (float) for chat/multiple-choice-questions.
        """
        raise NotImplementedError
